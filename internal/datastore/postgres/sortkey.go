package postgres

import (
	"crypto/sha256"
	"encoding/binary"
	"slices"

	"github.com/authzed/spicedb/pkg/spiceerrors"
)

// This file turns a Postgres transaction snapshot into bytes that sort.
//
// Snapshots are only PARTIALLY ordered: pgSnapshot.compare can return concurrent for two
// snapshots that are mutually uncomparable. Bytes are totally ordered. So the key cannot simply
// encode the order - it has to be a LINEAR EXTENSION of it, a total order that never contradicts
// the partial one:
//
//   - a.LessThan(b)      => key(a) sorts before key(b)
//   - a.Equal(b)         => key(a) and key(b) are byte-identical, including when the two
//     snapshots have different (xmin, xmax, xipList) triples
//   - a, b concurrent    => ordered deterministically, with no causal meaning
//
// Such a function exists because of one fact, which snapshot_totalorder_test.go establishes by
// brute force against txVisible rather than by restating the code: pgSnapshot.compare is exactly
// subset comparison of the sets of transactions each snapshot can see. LessThan is strict subset.
// A strict subset of a finite set is strictly smaller. Therefore the CARDINALITY of the visible
// set already orders every comparable pair, on its own, and everything after it in the key is a
// tiebreaker that only ever fires on pairs that have no order to contradict.
//
// That is what makes the key fixed width. The in-progress list does not need to be IN the key,
// only summarised by it, because the summary is never consulted for a pair that has a real order.

// sortKeyLength is the width of a Postgres revision sort key, in bytes. Fixed, whatever the
// snapshot looks like and however many transactions are in flight.
const sortKeyLength = 8 + sortKeyDigestLength

// sortKeyDigestLength is how much of the SHA-256 digest the key carries.
//
// 128 bits. The digest only ever separates snapshots that are concurrent, so a collision cannot
// invert a real order - it can only tie two snapshots that had no causal relationship to begin
// with. See the identity warning on AppendSortKey.
const sortKeyDigestLength = 16

// canonicalEncodingStackSize is the scratch buffer appendCanonicalEncoding is given before it has
// to reach for the heap. It covers the 16-in-flight case, which is far above typical.
const canonicalEncodingStackSize = 16 + 8*16

// AppendSortKey implements datastore.SortKeyRevision. Keys are sortKeyLength bytes.
//
// The key orders revisions in revision order for every pair that HAS an order, and imposes a
// deterministic order on pairs that do not. Specifically, for revisions a and b from the same
// datastore:
//
//   - a.LessThan(b) implies a's key sorts strictly before b's.
//   - a.Equal(b) implies the keys are byte-identical, even when the underlying snapshots are
//     written differently. snapshot 1:1: and snapshot 1:5:1,2,3,4 see exactly the same
//     transactions, are Equal, and key identically.
//   - Otherwise a and b are concurrent - neither happened before the other - and the key orders
//     them arbitrarily but consistently. Sorting by this key is stable across processes,
//     machines and SpiceDB versions.
//
// ⚠️ NEVER USE THIS KEY AS AN IDENTITY OR DEDUPLICATION KEY. It is not injective. Two concurrent
// revisions can produce the same key, because the digest is 128 bits of an unbounded input. A
// collision is harmless for ordering - the two revisions simply tie, and they had no order
// anyway - but code that treats equal keys as the same revision would silently merge two
// genuinely different ones. Use the revision itself for identity.
//
// The encoding is frozen. Keys may be stored durably, so no release may change these bytes;
// TestCompactKeyGoldenVectors pins them.
func (pr postgresRevision) AppendSortKey(dst []byte) []byte {
	return appendSnapshotSortKey(dst, pr.snapshot)
}

// appendSnapshotSortKey is the body of AppendSortKey, split out so the key can be computed for a
// bare pgSnapshot in tests without building a revision around it.
//
//	[8  bytes BE] visible-set cardinality          - orders every comparable pair, on its own
//	[16 bytes]    SHA-256(canonical encoding)[:16] - separates concurrent snapshots
//
// Cardinality is first because it is the part that carries the order; the digest is a tiebreaker
// below it. Big-endian because the key is compared as bytes.
func appendSnapshotSortKey(dst []byte, s pgSnapshot) []byte {
	var scratch [canonicalEncodingStackSize]byte
	digest := sha256.Sum256(appendCanonicalEncoding(scratch[:0], canonicalize(s)))

	dst = binary.BigEndian.AppendUint64(dst, visibleCardinality(s))
	return append(dst, digest[:sortKeyDigestLength]...)
}

// visibleCardinality returns how many transaction IDs the snapshot considers settled, which is
// |{t : s.txVisible(t)}|. Every txid at or above xmax is invisible, so the set is finite.
//
// Derived from txVisible, not assumed:
//
//   - txid <  xmin  -> visible, without the in-progress list ever being consulted
//   - txid >= xmax  -> invisible
//   - otherwise     -> visible iff not in xipList
//
// so the count is xmax minus the number of DISTINCT xipList entries that actually fall in
// [xmin, xmax). Those qualifiers are the whole difference between this and the obvious
// xmax - len(xipList), which is wrong three ways on a malformed snapshot: it counts entries below
// xmin that txVisible ignores, it counts duplicates twice, and it underflows to a huge uint64
// when the list is longer than the range. Postgres does not emit such snapshots and the mutators
// do not create them, but ScanText validates nothing beyond parsing.
// TestNaiveCardinalityFormulaBreaksOnIllFormedSnapshots pins each case.
func visibleCardinality(s pgSnapshot) uint64 {
	// Fast path. xipListIsClean has already established that every entry is distinct and inside
	// [xmin, xmax), which is exactly the condition under which each one hides one transaction.
	if xipListIsClean(s) {
		return s.xmax - spiceerrors.MustSafecast[uint64](len(s.xipList))
	}
	return s.xmax - spiceerrors.MustSafecast[uint64](len(cleanedXipList(s)))
}

// canonicalize returns the unique representative of the set of snapshots sharing a visible set,
// so that a digest taken over the result cannot disagree with Equal.
//
// This is required, not cosmetic. snap(1,1) and snap(1,5,1,2,3,4) are Equal under compare and
// have completely different triples; hashing the raw triple would give them different keys and
// break the Equal guarantee outright.
//
// The normal form drops the maximal run of in-progress transactions at the top of the range - a
// transaction nobody has seen settle is indistinguishable from one beyond xmax - and then pins
// xmin where markComplete pins it.
//
// The returned snapshot's xipList MAY ALIAS the receiver's. Callers must treat it as read-only.
func canonicalize(s pgSnapshot) pgSnapshot {
	xip := s.xipList
	if !xipListIsClean(s) {
		xip = cleanedXipList(s)
	}

	xmax := s.xmax
	n := len(xip)
	for n > 0 && xmax > 0 && xip[n-1] == xmax-1 {
		xmax--
		n--
	}
	xip = xip[:n]

	xmin := xmax
	if n > 0 {
		xmin = xip[0]
	} else {
		xip = nil
	}

	return pgSnapshot{xmin, xmax, xip}
}

// xipListIsClean reports whether the in-progress list is already sorted, distinct and wholly
// inside [xmin, xmax), which is what Postgres emits and what markComplete and markInProgress
// preserve. When it holds, canonicalize can subslice instead of allocating.
func xipListIsClean(s pgSnapshot) bool {
	for i, x := range s.xipList {
		if x < s.xmin || x >= s.xmax {
			return false
		}
		if i > 0 && x <= s.xipList[i-1] {
			return false
		}
	}
	return true
}

// cleanedXipList is the slow path for a snapshot that violates the invariant: it returns a sorted,
// deduplicated copy restricted to [xmin, xmax).
func cleanedXipList(s pgSnapshot) []uint64 {
	xip := make([]uint64, 0, len(s.xipList))
	for _, x := range s.xipList {
		if x < s.xmin || x >= s.xmax {
			continue
		}
		xip = append(xip, x)
	}
	slices.Sort(xip)
	return slices.Compact(xip)
}

// appendCanonicalEncoding appends an INJECTIVE byte encoding of a canonical snapshot to dst. It
// is the single definition of what the digest is taken over, and therefore part of the frozen
// wire format.
//
// Injectivity does NOT come from the count field, which is redundant - a claim an earlier version
// of this comment got wrong, caught by mutation testing. The encoding is a complete message
// rather than a prefix of a longer key, so two canonical forms differing only in how many entries
// they carry already produce byte strings of different lengths, and SHA-256 distinguishes those.
// The count is kept as cheap self-description for a durable format, not because anything rests
// on it.
//
// c must already be canonical. Passing a raw snapshot produces a well-formed but meaningless key.
func appendCanonicalEncoding(dst []byte, c pgSnapshot) []byte {
	dst = binary.BigEndian.AppendUint64(dst, c.xmax)
	dst = binary.BigEndian.AppendUint64(dst, spiceerrors.MustSafecast[uint64](len(c.xipList)))
	for _, x := range c.xipList {
		dst = binary.BigEndian.AppendUint64(dst, x)
	}
	return dst
}
