package postgres

// This file is an investigation, not a feature. It asks one question and tries hard to answer it
// with a falsifiable test rather than an argument:
//
//	Can a deterministic, stateless function from a pgSnapshot to bytes produce a LINEAR EXTENSION
//	of the partial order that pgSnapshot.compare implements?
//
// A linear extension is a total order that never contradicts the partial one:
//
//   - a.LessThan(b)    => key(a) <  key(b)    bytewise
//   - a.Equal(b)       => key(a) == key(b)    bytewise, including for snapshots that are Equal
//     but not textually identical
//   - a and b concurrent => key may order them either way, but deterministically, consistently,
//     and without creating a cycle
//
// Nothing here is production code. snapshot.go is untouched. The candidate key function lives in
// this file so that it can be attacked without anything depending on it.

import (
	"bytes"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"math"
	"math/bits"
	"math/rand/v2"
	"slices"
	"sync"
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/authzed/spicedb/pkg/spiceerrors"
)

// ---------------------------------------------------------------------------
// The production key
// ---------------------------------------------------------------------------

// compactSortKey is the PRODUCTION key, reached through the production entry point so that these
// properties are asserted about the code that actually ships rather than about a copy of it.
func compactSortKey(s pgSnapshot) []byte {
	return postgresRevision{snapshot: s}.AppendSortKey(nil)
}

// compactSortKeyWithDigestLength is compactSortKey with a shrinkable digest, so that collisions
// can be forced at a tiny width and the degradation mode observed directly rather than argued
// about. See TestTruncatedDigestOnlyEverCollidesIncomparablePairs.
//
// It necessarily restates the production key's layout, since the production function has no
// width knob. TestTruncatedDigestMatchesProductionAtFullWidth pins the two together so this copy
// cannot drift.
func compactSortKeyWithDigestLength(s pgSnapshot, digestLength int) []byte {
	digest := sha256.Sum256(appendCanonicalEncoding(nil, canonicalize(s)))

	key := make([]byte, 0, 8+digestLength)
	key = binary.BigEndian.AppendUint64(key, visibleCardinality(s))
	return append(key, digest[:digestLength]...)
}

// ---------------------------------------------------------------------------
// Independent ground truth, for small snapshots only
// ---------------------------------------------------------------------------

// bruteForceUniverse is the number of transaction IDs modelled exactly as a bitmask. Snapshots
// built by the small generators stay strictly below it.
const bruteForceUniverse = 40

// bruteForceVisibleSet returns the visible set as a bitmask by asking txVisible one txid at a
// time. It is deliberately the dumbest possible implementation: it is the oracle that the
// cardinality formula and the reduction of compare to subset comparison are checked against.
func bruteForceVisibleSet(t *testing.T, s pgSnapshot) uint64 {
	t.Helper()
	require.LessOrEqual(t, s.xmax, uint64(bruteForceUniverse), "snapshot too large for brute force")

	var mask uint64
	for txid := range uint64(bruteForceUniverse) {
		if s.txVisible(txid) {
			mask |= 1 << txid
		}
	}
	return mask
}

// ---------------------------------------------------------------------------
// Generators
// ---------------------------------------------------------------------------

// wellFormed reports whether a snapshot obeys the SAFETY invariant: sorted, distinct, and every
// in-progress entry inside [xmin, xmax). Postgres guarantees this, and markComplete /
// markInProgress preserve it (asserted by TestMutatorsPreserveWellFormedness).
//
// This deliberately does NOT require xmin to be pinned to the first in-progress entry. An xmin
// lower than it needs to be is harmless - txVisible returns true for everything below xmin, and
// with no in-progress entry down there the list lookup would have said visible anyway - and the
// existing test suite is full of such snapshots: snap(0,10), snap(1,3), snap(5,10). Pinning is a
// property of the canonical form, checked by isCanonical, not of a valid snapshot.
//
// The case that genuinely breaks things is an entry BELOW xmin, because txVisible ignores it
// (snapshot.go:334) while otherHasMoreInfo still consults it (snapshot.go:185). See
// TestIllFormedSnapshotsBreakCompareItself.
func wellFormed(s pgSnapshot) bool {
	if s.xmin > s.xmax {
		return false
	}
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

// isCanonical reports whether a snapshot is the unique representative of its equivalence class,
// which is what canonicalize returns.
func isCanonical(s pgSnapshot) bool {
	if !wellFormed(s) {
		return false
	}
	if len(s.xipList) == 0 {
		return s.xmin == s.xmax && s.xipList == nil
	}
	// xmin pinned to the first entry, and the top of the range actually visible.
	return s.xmin == s.xipList[0] && s.xipList[len(s.xipList)-1] != s.xmax-1
}

// makeWellFormed builds the canonical snapshot whose in-progress set is exactly inProgress,
// restricted to [0, xmax).
func makeWellFormed(xmax uint64, inProgress []uint64) pgSnapshot {
	xip := make([]uint64, 0, len(inProgress))
	for _, x := range inProgress {
		if x < xmax {
			xip = append(xip, x)
		}
	}
	slices.Sort(xip)
	xip = slices.Compact(xip)

	xmin := xmax
	if len(xip) > 0 {
		xmin = xip[0]
	} else {
		xip = nil
	}
	return pgSnapshot{xmin, xmax, xip}
}

// snapshotFromMask builds the well-formed snapshot whose in-progress set is the given bitmask
// below xmax. Enumerating masks is how the exhaustive sweep covers every shape in a small
// universe, including the ones random generation reaches only by luck.
func snapshotFromMask(xmax uint64, mask uint64) pgSnapshot {
	inProgress := make([]uint64, 0, bits.OnesCount64(mask))
	for i := range xmax {
		if mask&(1<<i) != 0 {
			inProgress = append(inProgress, i)
		}
	}
	return makeWellFormed(xmax, inProgress)
}

// fixtureSnapshots are every snapshot appearing in the existing snapshot_test.go, which encode
// knowledge about which shapes are tricky that random generation would reach only by accident:
// the large-xid cases, the original concurrency bug case, and the Equal-but-not-identical pairs.
func fixtureSnapshots() []pgSnapshot {
	return []pgSnapshot{
		snap(0, 0), snap(1, 1), snap(5, 5), snap(100, 100), snap(415, 415),
		snap(0, 4, 2), snap(0, 4, 2, 3),
		snap(10, 20, 12, 15, 18),
		snap(123, 456, 124, 126, 168),
		snap(1, 5, 1, 2, 3, 4),
		snap(5, 8, 5, 6, 7),
		snap(5, 10, 7), snap(3, 10, 3, 5, 7),
		snap(1, 3), snap(5, 10),
		snap(10, 10), snap(10, 15, 12), snap(10, 15),
		snap(8, 10, 8, 9),
		snap(5, 10, 5, 7), snap(5, 10, 5),
		snap(5, 10, 5, 6, 7, 8, 9), snap(5, 10, 5, 6, 7),
		snap(0, 5), snap(0, 10),
		snap(101, 101), snap(100, 104, 100, 101),
		snap(6, 8, 6, 7), snap(5, 9, 5, 8),
		snap(3, 5, 4), snap(2, 7, 2, 3, 6),
		snap(10, 14, 10, 12), snap(10, 14, 11, 13),
		snap(20, 20), snap(18, 25, 18, 19),
		snap(5, 12, 5, 8, 10), snap(5, 12, 6, 9, 11),
		snap(50, 50), snap(40, 60, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49),
		snap(3, 3), snap(3, 6, 3, 4, 5),
		snap(0, 1), snap(0, 2), snap(0, 2, 1),
		snap(5, 6), snap(5, 7),
		snap(0, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8),
		snap(0, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
		snap(840, 842, 840),

		// Large xids: these must not be ordered by anything that iterates the range.
		snap(1<<63, 1<<63), snap(2, 1<<63), snap(1<<62, 1<<62),
		snap(1<<61, 1<<63, 1<<61),
		snap(math.MaxUint64-1, math.MaxUint64-1),
		snap(math.MaxUint64-2, math.MaxUint64, math.MaxUint64-2, math.MaxUint64-1),

		// Results of the mutators, which is how snapshots are actually produced at runtime.
		snap(100, 100).markComplete(100),
		snap(100, 102, 100).markComplete(100),
		snap(0, 0).markComplete(5),
		snap(0, 6, 0, 1, 2, 3, 4).markInProgress(5),
		snap(5, 5).markInProgress(3),
	}
}

// adversarialSnapshots are pairs chosen to stress exactly the places the candidate could break:
// same cardinality but different visible sets, Equal-but-not-identical representations, empty
// and dense xipLists, and xmin == xmax.
func adversarialSnapshots() []pgSnapshot {
	out := make([]pgSnapshot, 0, 64)

	// Same cardinality, different visible set - forces the tiebreaker to do real work.
	for hidden := range uint64(8) {
		out = append(out, makeWellFormed(10, []uint64{hidden}))
	}

	// Every snapshot that differs from its neighbour in exactly one in-progress entry.
	for i := range uint64(9) {
		out = append(out, makeWellFormed(12, []uint64{i, i + 1}))
		out = append(out, makeWellFormed(12, []uint64{i, i + 2}))
	}

	// xmin == xmax, empty xipList, at a range of magnitudes.
	for _, x := range []uint64{0, 1, 2, 7, 64, 1 << 20, 1 << 40, 1 << 63} {
		out = append(out, pgSnapshot{x, x, nil})
	}

	// Entirely in-progress ranges, which canonicalize down to a much smaller snapshot.
	for xmax := range uint64(12) {
		all := make([]uint64, 0, xmax)
		for i := range xmax {
			all = append(all, i)
		}
		out = append(out, makeWellFormed(xmax, all))
	}

	// A long xipList, to make sure nothing is quadratic or truncating.
	long := make([]uint64, 0, 300)
	for i := range uint64(600) {
		if i%2 == 0 {
			long = append(long, i)
		}
	}
	out = append(out, makeWellFormed(600, long))
	out = append(out, makeWellFormed(601, long))

	return out
}

// randomSeed is fixed so that every run examines the same snapshots and a failure here is
// reproducible by rerunning the test unchanged.
const randomSeed = 0x5EED5C0DEBA5E

// randomSnapshots returns count pseudo-random well-formed snapshots. The universe is kept small
// on purpose: with xmax in [0, 24) distinct snapshots collide often, which is the only way the
// concurrent and Equal branches of compare get exercised at a useful rate.
func randomSnapshots(count int) []pgSnapshot {
	// A fixed seed is the whole point: a failure here must be reproducible by rerunning the test
	// unchanged, which a cryptographic source would defeat.
	rng := rand.New(rand.NewPCG(randomSeed, 0x9E3779B97F4A7C15)) //nolint:gosec

	out := make([]pgSnapshot, 0, count)
	for range count {
		xmax := rng.Uint64N(24)
		inProgress := make([]uint64, 0, xmax)
		for i := range xmax {
			if rng.Uint64N(3) == 0 {
				inProgress = append(inProgress, i)
			}
		}
		out = append(out, makeWellFormed(xmax, inProgress))
	}
	return out
}

// exhaustiveSmallSnapshots enumerates EVERY well-formed snapshot with xmax <= maxXmax. For
// maxXmax = 6 that is 1 + 2 + 4 + ... + 64 = 127 snapshots and 16129 ordered pairs, every shape
// in the universe rather than a sample of them.
func exhaustiveSmallSnapshots(maxXmax uint64) []pgSnapshot {
	out := []pgSnapshot{}
	for xmax := uint64(0); xmax <= maxXmax; xmax++ {
		for mask := range uint64(1) << xmax {
			out = append(out, snapshotFromMask(xmax, mask))
		}
	}
	return out
}

// allTestSnapshots is the full corpus every pairwise property is checked over.
func allTestSnapshots() []pgSnapshot {
	out := fixtureSnapshots()
	out = append(out, adversarialSnapshots()...)
	out = append(out, randomSnapshots(400)...)
	out = append(out, exhaustiveSmallSnapshots(6)...)
	return out
}

// ---------------------------------------------------------------------------
// Does the reduction hold? compare as subset comparison of visible sets
// ---------------------------------------------------------------------------

// TestCompareIsSubsetComparisonOfVisibleSets is the load-bearing claim. Everything else rests on
// it: if LessThan is exactly "strictly fewer transactions settled, and no disagreement", then
// cardinality is monotone and a linear extension exists. If it is anything subtler, the whole
// approach collapses.
//
// Checked against the brute-force visible set rather than against a restatement of
// otherHasMoreInfo (snapshot.go:182-215), so the test cannot be wrong in the same way the code is.
func TestCompareIsSubsetComparisonOfVisibleSets(t *testing.T) {
	snapshots := exhaustiveSmallSnapshots(6)
	snapshots = append(snapshots, randomSnapshots(200)...)

	var pairs int
	for _, a := range snapshots {
		if a.xmax > bruteForceUniverse {
			continue
		}
		for _, b := range snapshots {
			if b.xmax > bruteForceUniverse {
				continue
			}
			pairs++

			va := bruteForceVisibleSet(t, a)
			vb := bruteForceVisibleSet(t, b)

			aMinusB := va &^ vb
			bMinusA := vb &^ va

			var want comparisonResult
			switch {
			case aMinusB != 0 && bMinusA != 0:
				want = concurrent
			case bMinusA != 0:
				want = lt
			case aMinusB != 0:
				want = gt
			default:
				want = equal
			}

			require.Equal(t, want, a.compare(b),
				"compare(%s, %s): visible sets %012b vs %012b", a, b, va, vb)
		}
	}
	t.Logf("compare matched subset comparison on all %d pairs", pairs)
}

// TestCompareIsAStrictPartialOrder checks the precondition for a linear extension to exist at
// all. A relation with a cycle has no linear extension, and no key function could be written.
func TestCompareIsAStrictPartialOrder(t *testing.T) {
	snapshots := exhaustiveSmallSnapshots(5)
	snapshots = append(snapshots, fixtureSnapshots()...)
	snapshots = append(snapshots, randomSnapshots(120)...)

	for _, a := range snapshots {
		same := a
		require.True(t, a.Equal(same), "compare is not reflexive at %s", a)
		require.False(t, a.LessThan(same), "compare is not irreflexive at %s", a)
	}

	for _, a := range snapshots {
		for _, b := range snapshots {
			require.False(t, a.LessThan(b) && b.LessThan(a),
				"antisymmetry broken: %s and %s are each LessThan the other", a, b)
			require.Equal(t, a.LessThan(b), b.GreaterThan(a),
				"LessThan and GreaterThan disagree for %s, %s", a, b)
		}
	}

	var triples int
	for _, a := range snapshots {
		for _, b := range snapshots {
			if !a.LessThan(b) {
				continue
			}
			for _, c := range snapshots {
				if !b.LessThan(c) {
					continue
				}
				triples++
				require.True(t, a.LessThan(c),
					"transitivity broken: %s < %s < %s but not %s < %s", a, b, c, a, c)
			}
		}
	}
	t.Logf("partial order verified over %d snapshots, %d transitive triples", len(snapshots), triples)
}

// ---------------------------------------------------------------------------
// The cardinality formula
// ---------------------------------------------------------------------------

// TestVisibleCardinalityMatchesBruteForce pins the formula against the oracle.
func TestVisibleCardinalityMatchesBruteForce(t *testing.T) {
	snapshots := exhaustiveSmallSnapshots(6)
	snapshots = append(snapshots, randomSnapshots(200)...)

	for _, s := range snapshots {
		if s.xmax > bruteForceUniverse {
			continue
		}
		want := spiceerrors.MustSafecast[uint64](bits.OnesCount64(bruteForceVisibleSet(t, s)))
		require.Equal(t, want, visibleCardinality(s), "cardinality of %s", s)
	}
}

// TestNaiveCardinalityFormulaBreaksOnIllFormedSnapshots documents the limit of the obvious
// formula, xmax - len(xipList).
//
// For WELL-FORMED snapshots - the invariant Postgres guarantees and the mutators preserve - it is
// exactly right. For snapshots that violate the invariant it is wrong in two directions, and one
// of them underflows. This is why the candidate uses visibleCardinality instead.
func TestNaiveCardinalityFormulaBreaksOnIllFormedSnapshots(t *testing.T) {
	naive := func(s pgSnapshot) (uint64, bool) {
		if uint64(len(s.xipList)) > s.xmax {
			return 0, false // would underflow
		}
		return s.xmax - uint64(len(s.xipList)), true
	}

	t.Run("agrees for every well-formed snapshot", func(t *testing.T) {
		snapshots := exhaustiveSmallSnapshots(6)
		snapshots = append(snapshots, randomSnapshots(200)...)
		snapshots = append(snapshots, fixtureSnapshots()...)

		for _, s := range snapshots {
			if !wellFormed(s) {
				continue
			}
			got, ok := naive(s)
			require.True(t, ok, "naive formula underflowed on well-formed %s", s)
			require.Equal(t, visibleCardinality(s), got, "naive formula wrong for %s", s)
		}
	})

	t.Run("disagrees when an entry is below xmin", func(t *testing.T) {
		// txVisible returns true for txid < xmin before ever looking at xipList, so entry 3 hides
		// nothing. Every txid in [0,10) is visible.
		s := snap(5, 10, 3)
		require.False(t, wellFormed(s))
		require.EqualValues(t, 10, visibleCardinality(s))
		got, ok := naive(s)
		require.True(t, ok)
		require.EqualValues(t, 9, got, "naive formula is off by the out-of-range entry")
	})

	t.Run("disagrees on duplicates", func(t *testing.T) {
		// markComplete's own comment acknowledges a possibly "erroneously duplicate" entry
		// (snapshot.go:249).
		s := pgSnapshot{xmin: 3, xmax: 10, xipList: []uint64{3, 3, 4}}
		require.False(t, wellFormed(s))
		require.EqualValues(t, 8, visibleCardinality(s))
		got, ok := naive(s)
		require.True(t, ok)
		require.EqualValues(t, 7, got)
	})

	t.Run("underflows when the list is longer than the range", func(t *testing.T) {
		s := pgSnapshot{xmin: 0, xmax: 1, xipList: []uint64{5, 6, 7}}
		require.False(t, wellFormed(s))
		_, ok := naive(s)
		require.False(t, ok, "naive formula would underflow to a huge uint64")
		require.EqualValues(t, 1, visibleCardinality(s))
	})
}

// TestCanonicalizationMakesTheNaiveFormulaCorrect records a simplification found by mutation
// testing this file: replacing visibleCardinality(s) with the naive c.xmax - len(c.xipList) over
// the CANONICAL form breaks nothing, because canonicalize has already dropped out-of-range
// entries and duplicates.
//
// The two are interchangeable wherever the canonical form is already in hand. visibleCardinality
// is what production uses, because it states the intent directly and does not silently depend on
// canonicalize having run first.
func TestCanonicalizationMakesTheNaiveFormulaCorrect(t *testing.T) {
	snapshots := allTestSnapshots()
	snapshots = append(snapshots,
		snap(5, 10, 3), // entry below xmin
		pgSnapshot{xmin: 3, xmax: 10, xipList: []uint64{3, 3, 4}}, // duplicate
		pgSnapshot{xmin: 0, xmax: 1, xipList: []uint64{5, 6, 7}},  // entries above xmax
	)

	for _, s := range snapshots {
		c := canonicalize(s)
		require.Equal(t, visibleCardinality(s), c.xmax-uint64(len(c.xipList)),
			"canonical naive formula disagrees for %s (canonical %s)", s, c)
	}
}

// ---------------------------------------------------------------------------
// Canonicalization
// ---------------------------------------------------------------------------

// TestCanonicalizePreservesVisibleSetAndIsIdempotent checks the normal form does not change what
// a snapshot can see, and is a fixed point, which is what makes it a function of the equivalence
// class rather than of the representation.
func TestCanonicalizePreservesVisibleSetAndIsIdempotent(t *testing.T) {
	snapshots := exhaustiveSmallSnapshots(6)
	snapshots = append(snapshots, randomSnapshots(200)...)
	snapshots = append(snapshots, fixtureSnapshots()...)
	snapshots = append(snapshots, adversarialSnapshots()...)

	for _, s := range snapshots {
		c := canonicalize(s)
		require.True(t, s.Equal(c), "canonicalize changed the visible set: %s -> %s", s, c)
		require.True(t, isCanonical(c), "canonicalize produced a non-canonical snapshot: %s", c)
		require.Equal(t, c, canonicalize(c), "canonicalize is not idempotent at %s", s)

		// The normal form is a function of the equivalence class, so any two Equal snapshots in
		// the corpus must share it. That is the property the tiebreaker depends on.
		for _, other := range snapshots {
			if s.Equal(other) {
				require.Equal(t, c, canonicalize(other),
					"Equal snapshots %s and %s have different normal forms", s, other)
			}
		}
	}
}

// TestCanonicalizeCollapsesEqualButNotIdenticalSnapshots is the hazard that would sink any
// tiebreaker computed over the raw triple. These pairs are Equal under compare and have
// different (xmin, xmax, xipList).
func TestCanonicalizeCollapsesEqualButNotIdenticalSnapshots(t *testing.T) {
	pairs := []struct{ a, b pgSnapshot }{
		{snap(1, 1), snap(1, 5, 1, 2, 3, 4)},
		{snap(5, 5), snap(5, 8, 5, 6, 7)},
		{snap(3, 3), snap(3, 6, 3, 4, 5)},
		{snap(0, 0), snap(0, 4, 0, 1, 2, 3)},
		{snap(0, 2, 1), snap(0, 5, 1, 2, 3, 4)},
	}

	for _, p := range pairs {
		t.Run(fmt.Sprintf("%s==%s", p.a, p.b), func(t *testing.T) {
			require.True(t, p.a.Equal(p.b), "fixture is not actually Equal")
			require.NotEqual(t, p.a, p.b, "fixture is identical, so it proves nothing")
			require.Equal(t, canonicalize(p.a), canonicalize(p.b), "normal forms differ")
			require.Equal(t, compactSortKey(p.a), compactSortKey(p.b), "sort keys differ")
		})
	}
}

// ---------------------------------------------------------------------------
// Key hygiene
// ---------------------------------------------------------------------------

// TestCompactKeyIsDeterministicAndContextFree checks the key depends only on the snapshot's own
// fields, so it is stable as the database advances and does not depend on which other snapshots
// happen to be present.
func TestCompactKeyIsDeterministicAndContextFree(t *testing.T) {
	for _, s := range allTestSnapshots() {
		first := compactSortKey(s)
		for range 3 {
			require.Equal(t, first, compactSortKey(s), "key is not deterministic for %s", s)
		}

		// A different representation of the same snapshot, reached by a round trip through the
		// wire format, must key identically.
		text, err := s.TextValue()
		require.NoError(t, err)
		var reparsed pgSnapshot
		require.NoError(t, reparsed.ScanText(text))
		require.Equal(t, first, compactSortKey(reparsed),
			"key changed across a text round trip for %s", s)
	}
}

// TestCompactKeyIsPrefixFree checks that no key is a proper prefix of another, so a snapshot key
// can be one field of a longer composite key without the surrounding bytes reordering it. This
// is one of the guarantees datastore.SortKeyRevision.AppendSortKey makes.
//
// For a fixed-width key this is structural rather than incidental, and the test says so by
// asserting the width FIRST. A loop that only looks for prefixes would pass vacuously here - no
// key is ever shorter than another, so there is nothing for it to examine - and would keep
// passing if the key ever became variable width. Pinning the width is what gives it teeth.
func TestCompactKeyIsPrefixFree(t *testing.T) {
	snapshots := allTestSnapshots()

	for _, s := range snapshots {
		require.Len(t, compactSortKey(s), sortKeyLength,
			"prefix-freedom here rests on every key being the same width; %s broke that", s)
	}

	// Equal width makes a proper prefix impossible, so the only way to be a prefix is to be
	// equal, which only Equal snapshots may be.
	for _, a := range snapshots {
		for _, b := range snapshots {
			ka, kb := compactSortKey(a), compactSortKey(b)
			if bytes.HasPrefix(kb, ka) {
				require.Equal(t, ka, kb)
				require.False(t, a.LessThan(b) || a.GreaterThan(b),
					"keys of the comparable pair %s and %s are prefixes of one another", a, b)
			}
		}
	}
}

// TestCanonicalXmaxIsMonotoneButRawXmaxIsNot separates the two, because conflating them is what
// makes the naive "just sort by xmax" idea look broken when the real culprit is the missing
// canonicalization.
//
// This is a property of canonicalize, not of any particular key, which is why it outlived the
// variable-width key it was originally written for. It is also the reason the production key can
// put cardinality first without worrying: both fields move the same way, so the choice between
// them is free rather than load-bearing.
func TestCanonicalXmaxIsMonotoneButRawXmaxIsNot(t *testing.T) {
	t.Run("raw xmax is not monotone", func(t *testing.T) {
		low := snap(0, 10, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9) // sees nothing
		high := snap(0, 4, 2)                            // sees {0,1,3}
		require.True(t, low.LessThan(high))
		require.Greater(t, low.xmax, high.xmax, "raw xmax moves the wrong way")
	})

	t.Run("canonical xmax is monotone", func(t *testing.T) {
		snapshots := allTestSnapshots()
		var checked int
		for _, a := range snapshots {
			for _, b := range snapshots {
				if !a.LessThan(b) {
					continue
				}
				checked++
				require.LessOrEqual(t, canonicalize(a).xmax, canonicalize(b).xmax,
					"%s < %s but canonical xmax decreased", a, b)
			}
		}
		require.Positive(t, checked)
		t.Logf("canonical xmax non-decreasing across %d ordered pairs", checked)
	})

}

// ---------------------------------------------------------------------------
// Where the approach genuinely breaks
// ---------------------------------------------------------------------------

// TestIllFormedSnapshotsBreakCompareItself shows that an xipList entry below xmin makes compare
// inconsistent with the visible set it is supposed to be comparing, independently of any sort
// key. Two snapshots that see exactly the same transactions are reported as ordered.
//
// This is not reachable from Postgres, which emits xip entries only in [xmin, xmax), nor from
// markComplete / markInProgress, which is asserted separately by
// TestMutatorsPreserveWellFormedness. It is reachable from ScanText, because ScanText validates
// nothing beyond parsing (snapshot.go:64-100). It bounds how much any sort key could be trusted
// if a token's snapshot were ever attacker-supplied.
func TestIllFormedSnapshotsBreakCompareItself(t *testing.T) {
	ill := snap(5, 10, 3) // entry 3 is below xmin, so txVisible ignores it
	ok := snap(0, 10)     // no in-progress entries at all

	require.False(t, wellFormed(ill))
	require.True(t, wellFormed(ok))

	// They see exactly the same transactions.
	require.Equal(t, bruteForceVisibleSet(t, ill), bruteForceVisibleSet(t, ok))
	require.Equal(t, visibleCardinality(ill), visibleCardinality(ok))

	// And yet compare orders them, because otherHasMoreInfo consults the raw xipList
	// (snapshot.go:185) rather than asking whether this snapshot can see the entry.
	require.Equal(t, lt, ill.compare(ok),
		"expected the known inconsistency; if this now reports equal, compare was fixed")

	// So the sort key, which is computed from the visible set, cannot agree with compare here.
	require.Equal(t, compactSortKey(ill), compactSortKey(ok))
}

// TestMutatorsPreserveWellFormedness establishes that the ill-formed case above is unreachable
// from the code paths that actually produce snapshots at runtime.
func TestMutatorsPreserveWellFormedness(t *testing.T) {
	rng := rand.New(rand.NewPCG(randomSeed, 0xBF58476D1CE4E5B9)) //nolint:gosec // deterministic by design

	for _, start := range append(exhaustiveSmallSnapshots(5), randomSnapshots(150)...) {
		require.True(t, wellFormed(start), "generator produced an ill-formed snapshot %s", start)

		current := start
		for range 25 {
			txid := rng.Uint64N(32)
			if rng.Uint64N(2) == 0 {
				current = current.markComplete(txid)
				require.True(t, wellFormed(current),
					"markComplete(%d) on %s produced ill-formed %s", txid, start, current)
			} else {
				current = current.markInProgress(txid)
				require.True(t, wellFormed(current),
					"markInProgress(%d) on %s produced ill-formed %s", txid, start, current)
			}
		}
	}
}

// TestMarkCompleteIsMonotoneInTheCompactKey checks the thing a caller is most likely to assume: that
// committing another transaction moves the key forward, never backward.
func TestMarkCompleteIsMonotoneInTheCompactKey(t *testing.T) {
	for _, s := range append(exhaustiveSmallSnapshots(5), fixtureSnapshots()...) {
		if s.xmax > 1<<40 {
			continue // markComplete would materialize an enormous xipList
		}
		for txid := range uint64(12) {
			next := s.markComplete(txid)
			require.LessOrEqual(t, bytes.Compare(compactSortKey(s), compactSortKey(next)), 0,
				"markComplete(%d) moved the key backward: %s -> %s", txid, s, next)
		}
	}
}

// ---------------------------------------------------------------------------
// The compact key: 24 bytes, fixed width
// ---------------------------------------------------------------------------

// TestCardinalityAloneSeparatesEveryComparablePair is the single claim the fixed-width design
// rests on. If one comparable pair anywhere shares a cardinality, the digest would have to carry
// real ordering information, a 16-byte digest could not, and the design is dead.
//
// It follows from LessThan being exactly strict-subset-of-visible-sets
// (TestCompareIsSubsetComparisonOfVisibleSets): a strict subset of a finite set is strictly
// smaller. This asserts it over the corpus anyway, because a proof that rests on another test's
// result should be checked where it is used.
func TestCardinalityAloneSeparatesEveryComparablePair(t *testing.T) {
	snapshots := allTestSnapshots()

	cards := make([]uint64, len(snapshots))
	for i, s := range snapshots {
		cards[i] = visibleCardinality(s)
	}

	var ordered, equalPairs, concurrentPairs, concurrentSameCard int
	for i, a := range snapshots {
		for j, b := range snapshots {
			switch a.compare(b) {
			case lt:
				ordered++
				require.Less(t, cards[i], cards[j],
					"COUNTEREXAMPLE: %s < %s but cardinality %d is not less than %d",
					a, b, cards[i], cards[j])
			case gt:
				ordered++
				require.Greater(t, cards[i], cards[j],
					"COUNTEREXAMPLE: %s > %s but cardinality %d is not greater than %d",
					a, b, cards[i], cards[j])
			case equal:
				equalPairs++
				require.Equal(t, cards[i], cards[j],
					"COUNTEREXAMPLE: %s == %s but cardinalities %d and %d differ",
					a, b, cards[i], cards[j])
			case concurrent:
				concurrentPairs++
				if cards[i] == cards[j] {
					concurrentSameCard++
				}
			}
		}
	}

	t.Logf("ordered=%d equal=%d concurrent=%d, of which %d share a cardinality and so are "+
		"separated only by the digest", ordered, equalPairs, concurrentPairs, concurrentSameCard)

	require.Positive(t, ordered)
	require.Positive(t, concurrentSameCard,
		"no concurrent pair shares a cardinality, so the digest is never exercised and this "+
			"corpus cannot justify including it")
}

// TestCompactKeyIsALinearExtension runs the same three requirements as the variable-width key,
// against the 24-byte one.
func TestCompactKeyIsALinearExtension(t *testing.T) {
	snapshots := allTestSnapshots()
	keys := make([][]byte, len(snapshots))
	for i, s := range snapshots {
		keys[i] = compactSortKey(s)
	}

	var ltPairs, eqPairs, concurrentPairs int
	for i, a := range snapshots {
		for j, b := range snapshots {
			cmp := bytes.Compare(keys[i], keys[j])
			switch a.compare(b) {
			case lt:
				ltPairs++
				require.Negative(t, cmp, "LessThan but key not less:\n  %s -> %x\n  %s -> %x",
					a, keys[i], b, keys[j])
			case gt:
				require.Positive(t, cmp, "GreaterThan but key not greater:\n  %s -> %x\n  %s -> %x",
					a, keys[i], b, keys[j])
			case equal:
				eqPairs++
				require.Zero(t, cmp, "Equal but keys differ:\n  %s -> %x\n  %s -> %x",
					a, keys[i], b, keys[j])
			case concurrent:
				concurrentPairs++
			}
		}
	}

	t.Logf("seed=%#x snapshots=%d pairs=%d (lt/gt=%d equal=%d concurrent=%d)",
		randomSeed, len(snapshots), len(snapshots)*len(snapshots), ltPairs*2, eqPairs, concurrentPairs)
	require.Positive(t, ltPairs)
	require.Positive(t, concurrentPairs)
}

// TestCompactKeyIsFixedWidth is the property that makes this shippable on a public API field.
// The variable-width key reached 4020 bytes on the same input.
func TestCompactKeyIsFixedWidth(t *testing.T) {
	for _, s := range allTestSnapshots() {
		require.Len(t, compactSortKey(s), sortKeyLength, "width varied for %s", s)
	}

	inProgress := make([]uint64, 0, 500)
	for i := range uint64(1000) {
		if i%2 == 0 {
			inProgress = append(inProgress, i)
		}
	}
	big := makeWellFormed(1000, inProgress)

	require.Len(t, big.xipList, 500)
	require.Len(t, compactSortKey(big), sortKeyLength)
	t.Logf("%d in-progress transactions still key to %d bytes",
		len(big.xipList), len(compactSortKey(big)))
}

// TestCompactKeyEqualSnapshotsHashIdentically is the sharp case: Equal snapshots with different
// triples must produce byte-identical keys, which they do only because the digest is taken over
// the canonical form rather than the raw one.
func TestCompactKeyEqualSnapshotsHashIdentically(t *testing.T) {
	pairs := []struct{ a, b pgSnapshot }{
		{snap(1, 1), snap(1, 5, 1, 2, 3, 4)},
		{snap(5, 5), snap(5, 8, 5, 6, 7)},
		{snap(3, 3), snap(3, 6, 3, 4, 5)},
		{snap(0, 0), snap(0, 4, 0, 1, 2, 3)},
		{snap(0, 2, 1), snap(0, 5, 1, 2, 3, 4)},
		{snap(10, 10), snap(0, 10)},
	}

	for _, p := range pairs {
		t.Run(fmt.Sprintf("%s==%s", p.a, p.b), func(t *testing.T) {
			require.True(t, p.a.Equal(p.b), "fixture is not actually Equal")
			require.NotEqual(t, p.a, p.b, "fixture is identical, so it proves nothing")
			require.Equal(t, compactSortKey(p.a), compactSortKey(p.b))
		})
	}
}

// TestCompactKeyHasNoCollisionsInTheCorpus counts distinct canonical forms against distinct keys.
func TestCompactKeyHasNoCollisionsInTheCorpus(t *testing.T) {
	forms := map[string]pgSnapshot{}
	keys := map[string]pgSnapshot{}

	for _, s := range allTestSnapshots() {
		form := string(appendCanonicalEncoding(nil, canonicalize(s)))
		forms[form] = s

		key := string(compactSortKey(s))
		if prev, seen := keys[key]; seen {
			require.True(t, prev.Equal(s),
				"COLLISION between snapshots that are not Equal: %s and %s", prev, s)
		}
		keys[key] = s
	}

	require.Len(t, keys, len(forms),
		"distinct canonical forms collapsed into fewer keys, so a collision occurred")
	t.Logf("%d distinct canonical forms produced %d distinct keys, no collisions",
		len(forms), len(keys))
}

// TestTruncatedDigestOnlyEverCollidesIncomparablePairs is the heart of the collision argument,
// and the reason the trade is acceptable.
//
// Rather than observe that 128 bits is large and hope, it shrinks the digest to ONE BYTE to force
// collisions in a 635-snapshot corpus, then checks what actually breaks. The answer is that
// collisions appear only between snapshots that are Equal or concurrent. No comparable pair ever
// ties, at any digest width, because cardinality alone already separated them.
//
// So the failure mode of a hash collision is bounded by construction, not by probability: it can
// reorder two snapshots that have no causal relationship, and it can never invert a real order.
func TestTruncatedDigestOnlyEverCollidesIncomparablePairs(t *testing.T) {
	snapshots := allTestSnapshots()

	for _, digestLength := range []int{1, 2, 4, 16} {
		t.Run(fmt.Sprintf("%d-byte-digest", digestLength), func(t *testing.T) {
			keys := make([][]byte, len(snapshots))
			for i, s := range snapshots {
				keys[i] = compactSortKeyWithDigestLength(s, digestLength)
			}

			var collisions int
			for i, a := range snapshots {
				for j, b := range snapshots {
					cmp := bytes.Compare(keys[i], keys[j])

					switch a.compare(b) {
					case lt:
						require.Negative(t, cmp,
							"a real order was broken at digest width %d: %s < %s",
							digestLength, a, b)
					case gt:
						require.Positive(t, cmp,
							"a real order was broken at digest width %d: %s > %s",
							digestLength, a, b)
					case equal:
						require.Zero(t, cmp, "Equal snapshots must always tie")
					case concurrent:
						if cmp == 0 {
							collisions++
						}
					}
				}
			}

			t.Logf("digest width %2d bytes: %d concurrent pairs collided", digestLength, collisions)
			if digestLength == 1 {
				require.Positive(t, collisions,
					"a one-byte digest over %d snapshots should collide; if it does not, this "+
						"test is not demonstrating anything", len(snapshots))
			}
			if digestLength == sortKeyDigestLength {
				require.Zero(t, collisions, "no collision expected at the shipping width")
			}
		})
	}
}

// TestTruncatedDigestMatchesProductionAtFullWidth pins the test-only width-knob copy of the key
// layout against the production function, so the collision experiment cannot quietly drift away
// from the thing it is making claims about.
func TestTruncatedDigestMatchesProductionAtFullWidth(t *testing.T) {
	for _, s := range allTestSnapshots() {
		require.Equal(t,
			postgresRevision{snapshot: s}.AppendSortKey(nil),
			compactSortKeyWithDigestLength(s, sortKeyDigestLength),
			"the width-knob copy diverged from production for %s", s)
	}
}

// TestAppendSortKeyHonoursTheAppendContract checks the production entry point against the
// contract in pkg/datastore/datastore.go: write into dst, grow it like append, never replace it.
func TestAppendSortKeyHonoursTheAppendContract(t *testing.T) {
	for _, s := range allTestSnapshots() {
		rev := postgresRevision{snapshot: s}

		standalone := rev.AppendSortKey(nil)
		require.Len(t, standalone, sortKeyLength)

		prefix := []byte{0xDE, 0xAD, 0xBE, 0xEF}
		extended := rev.AppendSortKey(slices.Clone(prefix))
		require.Len(t, extended, len(prefix)+sortKeyLength)
		require.Equal(t, prefix, extended[:len(prefix)], "dst was not preserved")
		require.Equal(t, standalone, extended[len(prefix):], "appended key differs from standalone")

		// Appending twice to the same buffer yields the key twice, as the builtin append would.
		twice := rev.AppendSortKey(rev.AppendSortKey(nil))
		require.Len(t, twice, 2*sortKeyLength)
		require.Equal(t, standalone, twice[:sortKeyLength])
		require.Equal(t, standalone, twice[sortKeyLength:])

		// Into a buffer with spare capacity, so the append does not reallocate.
		roomy := make([]byte, 0, 4+sortKeyLength)
		roomy = append(roomy, prefix...)
		inPlace := rev.AppendSortKey(roomy)
		require.Equal(t, extended, inPlace)
	}
}

// TestAppendSortKeyIsConcurrencySafe exercises the "implementations keep no state" clause. The
// -race build is what makes this meaningful.
func TestAppendSortKeyIsConcurrencySafe(t *testing.T) {
	snapshots := allTestSnapshots()
	want := make([][]byte, len(snapshots))
	for i, s := range snapshots {
		want[i] = postgresRevision{snapshot: s}.AppendSortKey(nil)
	}

	var wg sync.WaitGroup
	for range 8 {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i, s := range snapshots {
				// Each goroutine owns its own dst, as the contract requires.
				if !bytes.Equal(want[i], postgresRevision{snapshot: s}.AppendSortKey(nil)) {
					t.Errorf("concurrent key mismatch for %s", s)
					return
				}
			}
		}()
	}
	wg.Wait()
}

// TestCompactKeyIsATotalPreorder states precisely what the compact key guarantees, which is
// weaker than the variable-width key's strict total order: ties are possible, but only between
// snapshots with no causal relationship, and the relation is still transitive, so sorting is
// well defined and cycle-free.
func TestCompactKeyIsATotalPreorder(t *testing.T) {
	snapshots := allTestSnapshots()
	keys := make([][]byte, len(snapshots))
	for i, s := range snapshots {
		keys[i] = compactSortKey(s)
	}

	for i, a := range snapshots {
		for j, b := range snapshots {
			ij := bytes.Compare(keys[i], keys[j])
			require.Equal(t, ij, -bytes.Compare(keys[j], keys[i]),
				"not antisymmetric for %s, %s", a, b)

			// A tie is permitted only where there is no order to contradict.
			if ij == 0 {
				require.False(t, a.LessThan(b) || a.GreaterThan(b),
					"keys tied for a comparable pair %s, %s", a, b)
			}
		}
	}

	sample := snapshots
	if len(sample) > 90 {
		sample = sample[:90]
	}
	for i := range sample {
		for j := range sample {
			if bytes.Compare(keys[i], keys[j]) > 0 {
				continue
			}
			for k := range sample {
				if bytes.Compare(keys[j], keys[k]) > 0 {
					continue
				}
				require.LessOrEqual(t, bytes.Compare(keys[i], keys[k]), 0,
					"preorder is not transitive: %s <= %s <= %s", sample[i], sample[j], sample[k])
			}
		}
	}
}

// TestCompactSortedOrderNeverContradictsThePartialOrder sorts the corpus by compact key and
// checks the sequence, which is what a caller actually does with it.
func TestCompactSortedOrderNeverContradictsThePartialOrder(t *testing.T) {
	sorted := slices.Clone(allTestSnapshots())
	slices.SortStableFunc(sorted, func(a, b pgSnapshot) int {
		return bytes.Compare(compactSortKey(a), compactSortKey(b))
	})

	for i := range sorted {
		for j := i + 1; j < len(sorted); j++ {
			require.False(t, sorted[i].GreaterThan(sorted[j]),
				"position %d holds %s which is GreaterThan %s at position %d",
				i, sorted[i], sorted[j], j)
		}
	}
	t.Logf("sorted %d snapshots by compact key, no inversion", len(sorted))
}

// TestCardinalityCannotOverflow answers whether eight bytes is enough at the top of the xid
// range.
//
// Cardinality is xmax minus the number of hidden transactions, both uint64, and hidden can never
// exceed xmax, so the result is in [0, MaxUint64] and the subtraction cannot underflow. The only
// way to exceed the field would be for xmax itself to exceed uint64, which it cannot: Postgres
// xid8 is a 64-bit FullTransactionId that does not wrap, which is the whole reason SpiceDB uses
// it rather than the wrapping 32-bit xid.
func TestCardinalityCannotOverflow(t *testing.T) {
	topOfRange := pgSnapshot{xmin: math.MaxUint64, xmax: math.MaxUint64}
	require.Equal(t, uint64(math.MaxUint64), visibleCardinality(topOfRange))
	require.Equal(t, []byte{0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff},
		compactSortKey(topOfRange)[:8], "the cardinality field saturates rather than wrapping")

	// One hidden transaction at the very top is one below the maximum, not a wrap to zero.
	nearMax := pgSnapshot{
		xmin:    math.MaxUint64 - 1,
		xmax:    math.MaxUint64,
		xipList: []uint64{math.MaxUint64 - 1},
	}
	require.Equal(t, uint64(math.MaxUint64)-1, visibleCardinality(nearMax))
	require.True(t, nearMax.LessThan(topOfRange))
	require.Negative(t, bytes.Compare(compactSortKey(nearMax), compactSortKey(topOfRange)))

	// And the empty snapshot sits at the bottom without underflowing.
	require.EqualValues(t, 0, visibleCardinality(snap(0, 0)))
	require.Equal(t, make([]byte, 8), compactSortKey(snap(0, 0))[:8])
}

// TestCompactKeyGoldenVectors pins exact bytes. Keys may be stored durably, so the encoding can
// never change once released; if a refactor alters the canonical form, the hash, the field order
// or the endianness, this test is what notices.
func TestCompactKeyGoldenVectors(t *testing.T) {
	vectors := []struct {
		snapshot pgSnapshot
		want     string
	}{
		{snap(0, 0), "0000000000000000374708fff7719dd5979ec875d56cd228"},
		{snap(1, 1), "0000000000000001783825822a6f9e62da2190e828e4c9d2"},
		// Equal to the line above despite a completely different triple, so byte-identical.
		{snap(1, 5, 1, 2, 3, 4), "0000000000000001783825822a6f9e62da2190e828e4c9d2"},
		{snap(0, 4, 2), "0000000000000003c5fc36f5fdc9e58c969a372f4d52a873"},
		{snap(123, 456, 124, 126, 168), "00000000000001c5eff36a835d00f1e06797bdf06bdcdcf7"},
		{snap(10, 20, 12, 15, 18), "0000000000000011598b48f943f01f513debcc8c1ecfff33"},
		// Top of the xid range: the cardinality field saturates at all-ones, it does not wrap.
		{
			pgSnapshot{xmin: math.MaxUint64, xmax: math.MaxUint64},
			"ffffffffffffffff60c69a3e87bf5c4f1e546bec45f26269",
		},
	}

	for _, v := range vectors {
		got := hex.EncodeToString(compactSortKey(v.snapshot))
		require.Equal(t, v.want, got, "golden vector changed for %s", v.snapshot)
		require.Len(t, got, sortKeyLength*2)
	}
}

// ---------------------------------------------------------------------------
// The revision level, which is where a ZedToken actually operates
// ---------------------------------------------------------------------------

// TestRevisionSortKeyIsALinearExtension lifts the result from pgSnapshot to postgresRevision.
// This is the level that matters: a ZedToken carries a revision, not a snapshot, and
// postgresRevision delegates all three comparisons to its snapshot (revisions.go:369-387).
func TestRevisionSortKeyIsALinearExtension(t *testing.T) {
	snapshots := allTestSnapshots()

	revisions := make([]postgresRevision, 0, len(snapshots))
	for i, s := range snapshots {
		// Deliberately varied, and deliberately anti-correlated with the snapshot order, to prove
		// the key ignores both fields.
		noise := spiceerrors.MustSafecast[uint64](len(snapshots) - i)
		revisions = append(revisions, postgresRevision{
			snapshot:                      s,
			optionalTxID:                  xid8{Uint64: noise, Valid: true},
			optionalInexactNanosTimestamp: noise,
		})
	}

	for _, a := range revisions {
		for _, b := range revisions {
			cmp := bytes.Compare(compactSortKey(a.snapshot), compactSortKey(b.snapshot))
			switch {
			case a.LessThan(b):
				require.Negative(t, cmp, "revision %s < %s", a.snapshot, b.snapshot)
			case a.GreaterThan(b):
				require.Positive(t, cmp, "revision %s > %s", a.snapshot, b.snapshot)
			case a.Equal(b):
				require.Zero(t, cmp, "revision %s == %s", a.snapshot, b.snapshot)
			}
		}
	}
}

// TestCommitTimestampCannotBeUsedAsATiebreaker answers whether the stored commit timestamp could
// stand in for, or break ties in, the sort key. It cannot, and the reason is structural rather
// than a matter of clock quality.
//
// postgresRevision.Equal compares ONLY the snapshot (revisions.go:369-372). Two revisions with
// the same snapshot and different timestamps are Equal, so a key that mixes in the timestamp
// gives them different keys and violates the Equal requirement outright.
func TestCommitTimestampCannotBeUsedAsATiebreaker(t *testing.T) {
	s := snap(5, 10, 5, 7)

	early := postgresRevision{snapshot: s, optionalInexactNanosTimestamp: 1_000}
	late := postgresRevision{snapshot: s, optionalInexactNanosTimestamp: 9_000_000}

	require.True(t, early.Equal(late),
		"revisions with the same snapshot must be Equal regardless of timestamp")
	require.False(t, early.LessThan(late))
	require.False(t, late.GreaterThan(early))

	// The snapshot-derived key respects that. A timestamp-derived one would not.
	require.Equal(t, compactSortKey(early.snapshot), compactSortKey(late.snapshot))

	timestampKey := func(pr postgresRevision) []byte {
		return binary.BigEndian.AppendUint64(nil, pr.optionalInexactNanosTimestamp)
	}
	require.NotEqual(t, timestampKey(early), timestampKey(late),
		"a timestamp key separates two Equal revisions, which is the violation")

	// And the timestamp is frequently absent altogether, so it could not be relied on even if it
	// were ordered: it is loaded only where a caller needs it (pgrevision.proto:19-23).
	bare := postgresRevision{snapshot: s}
	_, ok := bare.OptionalNanosTimestamp()
	require.False(t, ok, "timestamp is optional and routinely missing")
}

// TestZedTokenAlreadyCarriesEnoughToComputeTheKey is the finding that matters most for a
// non-Go client.
//
// A ZedToken's revision string is postgresRevision.String(), which is base64 of the
// PostgresRevision proto, which carries xmin, relative_xmax and relative_xips - the entire
// snapshot (revisions.go:392-456, pgrevision.proto:15-24). So the sort key is computable from
// the token alone, with no database round trip.
//
// What a client CANNOT do is know that it should parse the revision that way: the token records
// a datastore unique ID prefix but no datastore type, and every datastore encodes its revision
// string differently.
func TestZedTokenAlreadyCarriesEnoughToComputeTheKey(t *testing.T) {
	for _, s := range allTestSnapshots() {
		if !encodableAsRevisionProto(s) {
			continue // see TestRevisionStringPanicsOnExtremeXids
		}
		original := postgresRevision{snapshot: s}

		// This string is exactly what lands in DecodedZedToken.V1ZedToken.revision.
		encoded := original.String()

		parsedRaw, err := parseRevisionProto(encoded)
		require.NoError(t, err, "round trip failed for %s", s)
		parsed, ok := parsedRaw.(postgresRevision)
		require.True(t, ok)

		require.Equal(t, compactSortKey(s), compactSortKey(parsed.snapshot),
			"sort key not recoverable from the encoded revision for %s", s)
		require.True(t, original.Equal(parsed))
	}
}

// encodableAsRevisionProto reports whether a snapshot survives postgresRevision.String().
// MarshalBinary encodes xmax and the xip list as offsets from xmin in int64 fields
// (revisions.go:427-447), so anything that overflows a signed 64-bit value is rejected.
func encodableAsRevisionProto(s pgSnapshot) bool {
	if s.xmin > math.MaxInt64 || s.xmax > math.MaxInt64 {
		return false
	}
	for _, x := range s.xipList {
		if x > math.MaxInt64 {
			return false
		}
	}
	return true
}

// TestRevisionStringPanicsOnExtremeXids is an incidental pre-existing finding, not a property of
// any sort key.
//
// postgresRevision.String() panics rather than erroring for a snapshot whose xids exceed
// MaxInt64, because MarshalBinary uses MustBugf on the int64 cast (revisions.go:431). The
// existing snapshot_test.go already exercises snap(1<<63, 1<<63) against compare, so the value
// is considered legitimate at the snapshot layer but is unrepresentable one layer up.
//
// It is not reachable in practice - a Postgres xid8 counter would have to pass 2^63 - but it
// matters to this investigation because any scheme that computes a sort key at token-issue time
// runs on exactly this code path.
//
// WHAT IT WOULD TAKE TO FIX, for whoever implements the sort key. MarshalBinary encodes xmax and
// each xip as a signed offset from xmin so that protobuf varints stay short for the common case
// where the three are close together. The offsets are genuinely signed - an xip is always above
// xmin, but relative_xmax is computed as xmax-xmin and the proto field is int64 - so the fix is
// not simply widening to uint64. The honest options are to add a uint64 absolute-value field
// alongside the relative one and prefer it when the relative encoding would overflow, or to
// change MarshalBinary to return its error instead of calling MustBugf and let String() surface
// it, which changes a widely used signature. Either is a self-contained change with its own wire
// or API consequences, so it belongs in a separate PR from the sort key, and it is not urgent:
// no deployment can reach xid 2^63. The reason to track it at all is that the sort key moves
// this code from "called when someone serialises a revision" to "called on every token issue",
// so a panic there becomes a request-path panic rather than a debugging inconvenience.
func TestRevisionStringPanicsOnExtremeXids(t *testing.T) {
	extreme := postgresRevision{snapshot: snap(1<<63, 1<<63)}
	require.Panics(t, func() { _ = extreme.String() },
		"if this no longer panics, MarshalBinary learned to handle large xids and the skip in "+
			"TestZedTokenAlreadyCarriesEnoughToComputeTheKey can be removed")

	// The sort key itself has no such limit: it is pure uint64 arithmetic.
	require.NotPanics(t, func() { _ = compactSortKey(snap(1<<63, 1<<63)) })
	require.NotPanics(t, func() {
		_ = compactSortKey(snap(math.MaxUint64-2, math.MaxUint64, math.MaxUint64-2))
	})
}

// FuzzCompactSortKeyIsALinearExtension is the fuzz target for the shippable 24-byte key.
//
// Note the concurrent case is deliberately NOT asserted to differ. The compact key is a total
// PREORDER: two concurrent snapshots are permitted to collide, and asserting otherwise would be
// asserting that SHA-256 never collides. What must hold, and is asserted, is that no COMPARABLE
// pair ever ties or inverts.
func FuzzCompactSortKeyIsALinearExtension(f *testing.F) {
	f.Add(uint64(4), uint64(0b0100), uint64(4), uint64(0b1100))
	f.Add(uint64(0), uint64(0), uint64(1), uint64(0))
	f.Add(uint64(5), uint64(0b11111), uint64(5), uint64(0))
	f.Add(uint64(12), uint64(0b101010101010), uint64(12), uint64(0b010101010101))
	f.Add(uint64(40), uint64(1)<<39, uint64(40), uint64(0))

	f.Fuzz(func(t *testing.T, xmaxA, maskA, xmaxB, maskB uint64) {
		xmaxA %= bruteForceUniverse + 1
		xmaxB %= bruteForceUniverse + 1

		a := snapshotFromMask(xmaxA, maskA)
		b := snapshotFromMask(xmaxB, maskB)

		keyA, keyB := compactSortKey(a), compactSortKey(b)
		require.Len(t, keyA, sortKeyLength)
		require.Len(t, keyB, sortKeyLength)

		cmp := bytes.Compare(keyA, keyB)
		switch a.compare(b) {
		case lt:
			require.Negative(t, cmp, "LessThan but key not less: %s, %s", a, b)
			require.Less(t, visibleCardinality(a), visibleCardinality(b),
				"cardinality failed to separate a comparable pair: %s, %s", a, b)
		case gt:
			require.Positive(t, cmp, "GreaterThan but key not greater: %s, %s", a, b)
			require.Greater(t, visibleCardinality(a), visibleCardinality(b),
				"cardinality failed to separate a comparable pair: %s, %s", a, b)
		case equal:
			require.Zero(t, cmp, "Equal but keys differ: %s, %s", a, b)
		}
	})
}
