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
	"encoding/binary"
	"fmt"
	"math"
	"math/bits"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// The candidate under test
// ---------------------------------------------------------------------------

// visibleCardinality returns how many transaction IDs the snapshot considers settled, which is
// |{t : s.txVisible(t)}|. Every txid >= xmax is invisible, so the set is finite.
//
// Derived from txVisible (snapshot.go:332-342), not assumed:
//
//   - txid <  xmin            -> visible    (xmin of them)
//   - txid >= xmax            -> invisible
//   - otherwise               -> visible iff not in xipList
//
// so the count is xmax minus the number of DISTINCT xipList entries that actually fall in
// [xmin, xmax). The qualifier matters. txVisible ignores an xipList entry below xmin - it returns
// true before ever consulting the list - and it ignores one at or above xmax. The naive
// xmax - len(xipList) counts both, and double-counts duplicates.
func visibleCardinality(s pgSnapshot) uint64 {
	sorted := slices.Clone(s.xipList)
	slices.Sort(sorted)

	var hidden uint64
	var prev uint64
	havePrev := false
	for _, x := range sorted {
		if x < s.xmin || x >= s.xmax {
			continue // txVisible never consults the list for these
		}
		if havePrev && x == prev {
			continue // a duplicate hides nothing extra
		}
		hidden++
		prev = x
		havePrev = true
	}

	return s.xmax - hidden
}

// canonicalize returns the unique representative of the equivalence class of snapshots with the
// same visible set, so that a tiebreaker computed over the triple cannot disagree with Equal.
//
// This is required, not cosmetic: snap(1,1) and snap(1,5,1,2,3,4) are Equal under compare
// (asserted by the existing TestCompare, snapshot_test.go:110) and have completely different
// triples. Any tiebreaker over the raw (xmin, xmax, xipList) breaks the Equal requirement.
//
// The normal form drops the maximal suffix of in-progress transactions at the top of the range -
// a transaction nobody has seen settle is indistinguishable from one beyond xmax - and then pins
// xmin where markComplete does (snapshot.go:257-262).
func canonicalize(s pgSnapshot) pgSnapshot {
	xip := make([]uint64, 0, len(s.xipList))
	for _, x := range s.xipList {
		if x < s.xmin || x >= s.xmax {
			continue
		}
		xip = append(xip, x)
	}
	slices.Sort(xip)
	xip = slices.Compact(xip)

	xmax := s.xmax
	for len(xip) > 0 && xmax > 0 && xip[len(xip)-1] == xmax-1 {
		xmax--
		xip = xip[:len(xip)-1]
	}

	xmin := xmax
	if len(xip) > 0 {
		xmin = xip[0]
	} else {
		xip = nil
	}

	return pgSnapshot{xmin, xmax, xip}
}

// snapshotSortKey is THE CANDIDATE. Bytes that are claimed to sort as a linear extension of
// compare.
//
//	[8 bytes BE] visible-set cardinality   - monotone, carries the order
//	[8 bytes BE] canonical xmax            - also monotone, breaks ties
//	[4 bytes BE] canonical len(xipList)    - redundant; see below
//	[8 bytes BE] * N  canonical xipList    - breaks the remaining ties, ascending
//
// Two notes, both established by mutation testing this file rather than by reasoning:
//
// Cardinality does NOT have to come first. Canonical xmax is monotone too - canonicalize pins it
// to one past the highest settled transaction, so a strict subset of visible transactions can
// only lower it - and swapping the first two fields still yields a valid linear extension
// (TestCanonicalXmaxIsMonotoneButRawXmaxIsNot). RAW xmax is not monotone, which is the easy
// mistake: snap(0,10,0..9) sees nothing and is LessThan snap(0,4,2), yet has the larger raw xmax.
// It canonicalizes to snap(0,0), so the trap is avoided by canonicalizing, not by field order.
//
// The length prefix is redundant. Cardinality is canonical xmax minus the entry count, so any two
// of the three determine the third, and keys that agree on the first sixteen bytes already have
// equal length. It is kept because a real implementation should not make a reader derive its
// framing, and because removing it makes prefix-freedom depend on an invariant three fields away
// (TestKeyLengthIsDeterminedByItsHeader).
func snapshotSortKey(s pgSnapshot) []byte {
	c := canonicalize(s)

	if len(c.xipList) > math.MaxUint32 {
		panic("xipList too long to encode")
	}

	key := make([]byte, 0, 20+8*len(c.xipList))
	key = binary.BigEndian.AppendUint64(key, visibleCardinality(s))
	key = binary.BigEndian.AppendUint64(key, c.xmax)
	key = binary.BigEndian.AppendUint32(key, uint32(len(c.xipList)))
	for _, x := range c.xipList {
		key = binary.BigEndian.AppendUint64(key, x)
	}

	return key
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
	out := []pgSnapshot{}

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
	rng := rand.New(rand.NewPCG(randomSeed, 0x9E3779B97F4A7C15))

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
		require.True(t, a.Equal(a), "compare is not reflexive at %s", a)
		require.False(t, a.LessThan(a), "compare is not irreflexive at %s", a)
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
		want := uint64(bits.OnesCount64(bruteForceVisibleSet(t, s)))
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
			require.Equal(t, snapshotSortKey(p.a), snapshotSortKey(p.b), "sort keys differ")
		})
	}
}

// ---------------------------------------------------------------------------
// The three required properties
// ---------------------------------------------------------------------------

// TestSortKeyIsALinearExtension is the headline test. Every ordered pair in the corpus is checked
// against all three requirements.
func TestSortKeyIsALinearExtension(t *testing.T) {
	snapshots := allTestSnapshots()
	keys := make([][]byte, len(snapshots))
	for i, s := range snapshots {
		keys[i] = snapshotSortKey(s)
	}

	var pairs, ltPairs, eqPairs, concurrentPairs int
	for i, a := range snapshots {
		for j, b := range snapshots {
			pairs++
			cmp := bytes.Compare(keys[i], keys[j])

			switch a.compare(b) {
			case lt:
				ltPairs++
				require.Negative(t, cmp,
					"LessThan but key not less:\n  a = %s key=%x\n  b = %s key=%x", a, keys[i], b, keys[j])
			case gt:
				require.Positive(t, cmp,
					"GreaterThan but key not greater:\n  a = %s key=%x\n  b = %s key=%x", a, keys[i], b, keys[j])
			case equal:
				eqPairs++
				require.Zero(t, cmp,
					"Equal but keys differ:\n  a = %s key=%x\n  b = %s key=%x", a, keys[i], b, keys[j])
			case concurrent:
				concurrentPairs++
				require.NotZero(t, cmp,
					"concurrent but keys collide, so the order is not total:\n  a = %s\n  b = %s", a, b)
			default:
				t.Fatalf("unexpected comparison result for %s and %s", a, b)
			}
		}
	}

	t.Logf("seed=%#x snapshots=%d pairs=%d (lt/gt=%d equal=%d concurrent=%d)",
		randomSeed, len(snapshots), pairs, ltPairs*2, eqPairs, concurrentPairs)

	require.Positive(t, ltPairs, "corpus produced no ordered pairs, the test proves nothing")
	require.Positive(t, concurrentPairs, "corpus produced no concurrent pairs, the hard case is untested")
	require.Positive(t, eqPairs, "corpus produced no Equal pairs")
}

// TestSortKeyOrderIsAStrictTotalOrder checks the derived order is well behaved in its own right,
// independent of compare: irreflexive, antisymmetric, transitive, and total once Equal snapshots
// are identified.
func TestSortKeyOrderIsAStrictTotalOrder(t *testing.T) {
	snapshots := allTestSnapshots()
	keys := make([][]byte, len(snapshots))
	for i, s := range snapshots {
		keys[i] = snapshotSortKey(s)
	}

	for i := range snapshots {
		require.Zero(t, bytes.Compare(keys[i], keys[i]), "key order is not reflexive at %s", snapshots[i])
	}

	for i, a := range snapshots {
		for j, b := range snapshots {
			ij := bytes.Compare(keys[i], keys[j])
			ji := bytes.Compare(keys[j], keys[i])
			require.Equal(t, ij, -ji, "key order is not antisymmetric for %s, %s", a, b)
			require.Equal(t, ij == 0, a.Equal(b),
				"keys tie iff Equal is violated for %s, %s", a, b)
		}
	}

	// Transitivity of a lexicographic byte order is a mathematical fact, but a cycle would show
	// up here if the key were not a pure function, so a bounded sweep is still worth running.
	sample := snapshots
	if len(sample) > 90 {
		sample = sample[:90]
	}
	for i := range sample {
		for j := range sample {
			if bytes.Compare(keys[i], keys[j]) >= 0 {
				continue
			}
			for k := range sample {
				if bytes.Compare(keys[j], keys[k]) >= 0 {
					continue
				}
				require.Negative(t, bytes.Compare(keys[i], keys[k]),
					"key order is not transitive: %s < %s < %s", sample[i], sample[j], sample[k])
			}
		}
	}
}

// TestSortedOrderNeverContradictsThePartialOrder sorts the whole corpus by key and then checks
// every pair in the resulting sequence. This is the property a caller actually depends on, and it
// is where a bad tiebreaker surfaces as an element appearing before something it is GreaterThan.
func TestSortedOrderNeverContradictsThePartialOrder(t *testing.T) {
	snapshots := allTestSnapshots()

	sorted := slices.Clone(snapshots)
	slices.SortStableFunc(sorted, func(a, b pgSnapshot) int {
		return bytes.Compare(snapshotSortKey(a), snapshotSortKey(b))
	})

	for i := range sorted {
		for j := i + 1; j < len(sorted); j++ {
			require.False(t, sorted[i].GreaterThan(sorted[j]),
				"sorted position %d holds %s which is GreaterThan %s at position %d",
				i, sorted[i], sorted[j], j)
		}
	}
	t.Logf("sorted %d snapshots, no inversion against the partial order", len(sorted))
}

// ---------------------------------------------------------------------------
// Key hygiene
// ---------------------------------------------------------------------------

// TestSortKeyIsDeterministicAndContextFree checks the key depends only on the snapshot's own
// fields, so it is stable as the database advances and does not depend on which other snapshots
// happen to be present.
func TestSortKeyIsDeterministicAndContextFree(t *testing.T) {
	for _, s := range allTestSnapshots() {
		first := snapshotSortKey(s)
		for range 3 {
			require.Equal(t, first, snapshotSortKey(s), "key is not deterministic for %s", s)
		}

		// A different representation of the same snapshot, reached by a round trip through the
		// wire format, must key identically.
		text, err := s.TextValue()
		require.NoError(t, err)
		var reparsed pgSnapshot
		require.NoError(t, reparsed.ScanText(text))
		require.Equal(t, first, snapshotSortKey(reparsed),
			"key changed across a text round trip for %s", s)
	}
}

// TestSortKeyIsPrefixFree checks that no key is a prefix of another, so a snapshot key can be one
// field of a longer composite key without the surrounding bytes reordering it. This is the same
// guarantee datastore.SortKeyRevision.AppendSortKey makes.
func TestSortKeyIsPrefixFree(t *testing.T) {
	snapshots := allTestSnapshots()
	for _, a := range snapshots {
		for _, b := range snapshots {
			ka, kb := snapshotSortKey(a), snapshotSortKey(b)
			if len(ka) >= len(kb) {
				continue
			}
			require.False(t, bytes.HasPrefix(kb, ka),
				"key of %s is a proper prefix of key of %s", a, b)
		}
	}
}

// TestCanonicalXmaxIsMonotoneButRawXmaxIsNot separates the two, because conflating them is what
// makes the naive "just sort by xmax" idea look broken when the real culprit is the missing
// canonicalization.
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

	t.Run("swapping the first two fields is still a linear extension", func(t *testing.T) {
		// The alternative key, to show the design has slack rather than being load-bearing.
		altKey := func(s pgSnapshot) []byte {
			c := canonicalize(s)
			key := binary.BigEndian.AppendUint64(nil, c.xmax)
			key = binary.BigEndian.AppendUint64(key, visibleCardinality(s))
			for _, x := range c.xipList {
				key = binary.BigEndian.AppendUint64(key, x)
			}
			return key
		}

		snapshots := allTestSnapshots()
		for _, a := range snapshots {
			for _, b := range snapshots {
				cmp := bytes.Compare(altKey(a), altKey(b))
				switch a.compare(b) {
				case lt:
					require.Negative(t, cmp, "%s < %s", a, b)
				case gt:
					require.Positive(t, cmp, "%s > %s", a, b)
				case equal:
					require.Zero(t, cmp, "%s == %s", a, b)
				case concurrent:
					require.NotZero(t, cmp, "%s ~ %s", a, b)
				}
			}
		}
	})
}

// TestKeyLengthIsDeterminedByItsHeader is why prefix-freedom holds. Two keys sharing their first
// sixteen bytes necessarily have the same length, so neither can be a proper prefix of the other.
func TestKeyLengthIsDeterminedByItsHeader(t *testing.T) {
	byHeader := map[string]int{}
	for _, s := range allTestSnapshots() {
		key := snapshotSortKey(s)
		header := string(key[:16])
		if seen, ok := byHeader[header]; ok {
			require.Equal(t, seen, len(key),
				"two keys share a header but differ in length, prefix-freedom is not structural")
		}
		byHeader[header] = len(key)
	}
}

// TestSortKeySizeIsUnbounded records the practical cost, which is the main argument against
// embedding this key in a ZedToken. The key carries the canonical xipList, so it grows with the
// number of concurrently open write transactions.
func TestSortKeySizeIsUnbounded(t *testing.T) {
	require.Len(t, snapshotSortKey(snap(5, 5)), 20, "the empty case is small")

	inProgress := make([]uint64, 0, 500)
	for i := range uint64(1000) {
		if i%2 == 0 {
			inProgress = append(inProgress, i)
		}
	}
	big := makeWellFormed(1000, inProgress)
	key := snapshotSortKey(big)
	require.Len(t, key, 20+8*len(big.xipList))
	t.Logf("a snapshot with %d in-progress transactions produces a %d byte key",
		len(big.xipList), len(key))
	require.Greater(t, len(key), 4000, "unbounded growth is the point of this test")
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
	require.Equal(t, snapshotSortKey(ill), snapshotSortKey(ok))
}

// TestMutatorsPreserveWellFormedness establishes that the ill-formed case above is unreachable
// from the code paths that actually produce snapshots at runtime.
func TestMutatorsPreserveWellFormedness(t *testing.T) {
	rng := rand.New(rand.NewPCG(randomSeed, 0xBF58476D1CE4E5B9))

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

// TestMarkCompleteIsMonotoneInTheSortKey checks the thing a caller is most likely to assume: that
// committing another transaction moves the key forward, never backward.
func TestMarkCompleteIsMonotoneInTheSortKey(t *testing.T) {
	for _, s := range append(exhaustiveSmallSnapshots(5), fixtureSnapshots()...) {
		if s.xmax > 1<<40 {
			continue // markComplete would materialize an enormous xipList
		}
		for txid := range uint64(12) {
			next := s.markComplete(txid)
			require.LessOrEqual(t, bytes.Compare(snapshotSortKey(s), snapshotSortKey(next)), 0,
				"markComplete(%d) moved the key backward: %s -> %s", txid, s, next)
		}
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
		revisions = append(revisions, postgresRevision{
			snapshot: s,
			// Deliberately varied, and deliberately not correlated with the snapshot order, to
			// prove the key ignores them.
			optionalTxID:                  xid8{Uint64: uint64(len(snapshots) - i), Valid: true},
			optionalInexactNanosTimestamp: uint64(len(snapshots) - i),
		})
	}

	for _, a := range revisions {
		for _, b := range revisions {
			cmp := bytes.Compare(snapshotSortKey(a.snapshot), snapshotSortKey(b.snapshot))
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
	require.Equal(t, snapshotSortKey(early.snapshot), snapshotSortKey(late.snapshot))

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

		require.Equal(t, snapshotSortKey(s), snapshotSortKey(parsed.snapshot),
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
func TestRevisionStringPanicsOnExtremeXids(t *testing.T) {
	extreme := postgresRevision{snapshot: snap(1<<63, 1<<63)}
	require.Panics(t, func() { _ = extreme.String() },
		"if this no longer panics, MarshalBinary learned to handle large xids and the skip in "+
			"TestZedTokenAlreadyCarriesEnoughToComputeTheKey can be removed")

	// The sort key itself has no such limit: it is pure uint64 arithmetic.
	require.NotPanics(t, func() { _ = snapshotSortKey(snap(1<<63, 1<<63)) })
	require.NotPanics(t, func() {
		_ = snapshotSortKey(snap(math.MaxUint64-2, math.MaxUint64, math.MaxUint64-2))
	})
}

// ---------------------------------------------------------------------------
// Fuzzing
// ---------------------------------------------------------------------------

// FuzzSortKeyIsALinearExtension is the same three properties, driven by the fuzzer rather than a
// fixed corpus, so that `go test -fuzz` can keep attacking the candidate after this lands.
func FuzzSortKeyIsALinearExtension(f *testing.F) {
	f.Add(uint64(4), uint64(0b0100), uint64(4), uint64(0b1100))
	f.Add(uint64(0), uint64(0), uint64(1), uint64(0))
	f.Add(uint64(5), uint64(0b11111), uint64(5), uint64(0))
	f.Add(uint64(12), uint64(0b101010101010), uint64(12), uint64(0b010101010101))
	f.Add(uint64(40), uint64(1)<<39, uint64(40), uint64(0))

	f.Fuzz(func(t *testing.T, xmaxA, maskA, xmaxB, maskB uint64) {
		// Keep the universe inside a uint64 bitmask so the snapshots stay buildable.
		xmaxA %= bruteForceUniverse + 1
		xmaxB %= bruteForceUniverse + 1

		a := snapshotFromMask(xmaxA, maskA)
		b := snapshotFromMask(xmaxB, maskB)

		require.True(t, wellFormed(a))
		require.True(t, wellFormed(b))

		cmp := bytes.Compare(snapshotSortKey(a), snapshotSortKey(b))
		switch a.compare(b) {
		case lt:
			require.Negative(t, cmp, "LessThan but key not less: %s, %s", a, b)
		case gt:
			require.Positive(t, cmp, "GreaterThan but key not greater: %s, %s", a, b)
		case equal:
			require.Zero(t, cmp, "Equal but keys differ: %s, %s", a, b)
		case concurrent:
			require.NotZero(t, cmp, "concurrent but keys collide: %s, %s", a, b)
		}
	})
}
