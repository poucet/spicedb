package postgres

// Shared snapshot corpus, plus the properties of compare and canonicalize that the dominance
// tests in dominance_test.go rest on.
//
// Two facts are established here, and everything about causal comparison depends on them:
//
//  1. pgSnapshot.compare is exactly SUBSET COMPARISON of the sets of transactions each snapshot
//     can see. That is what makes "a dominates b" the same question as "visible(b) is a subset
//     of visible(a)", and therefore what makes Dominates implementable at all.
//  2. canonicalize is a function of the equivalence class, not of the representation. Two
//     snapshots that are Equal with different (xmin, xmax, xipList) triples share a normal form.
//     VisibleSet projects the canonical snapshot, so without this a remote caller would report
//     two Equal revisions as concurrent.
//
// Both are checked against a brute-force oracle built from txVisible rather than against a
// restatement of otherHasMoreInfo, so a test cannot be wrong in the same way the code is.
//
// ⚠️ No test here may iterate densely over a range bounded by xmax. The corpus deliberately
// contains snapshots at xmax = 1<<63 and above, so a loop over [0, xmax) does not merely run
// slowly, it does not terminate. Probe boundaries instead.

import (
	"fmt"
	"math"
	"math/bits"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/stretchr/testify/require"
)

// ---------------------------------------------------------------------------
// Independent ground truth, for small snapshots only
// ---------------------------------------------------------------------------

// bruteForceUniverse is the number of transaction IDs modelled exactly as a bitmask. Snapshots
// built by the small generators stay strictly below it.
const bruteForceUniverse = 40

// bruteForceVisibleSet returns the visible set as a bitmask by asking txVisible one txid at a
// time. It is deliberately the dumbest possible implementation: it is the oracle that the
// reduction of compare to subset comparison is checked against.
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

// adversarialSnapshots are pairs chosen to stress exactly the places dominance could break:
// same cardinality but different visible sets, Equal-but-not-identical representations, empty
// and dense xipLists, and xmin == xmax.
func adversarialSnapshots() []pgSnapshot {
	out := make([]pgSnapshot, 0, 64)

	// Same cardinality, different visible set - the shape that is concurrent rather than ordered.
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
// compare as subset comparison of visible sets
// ---------------------------------------------------------------------------

// TestCompareIsSubsetComparisonOfVisibleSets is the load-bearing claim for causal comparison.
// Dominance is defined as "visible(other) is a subset of visible(receiver)", and Dominates is
// implemented by routing through compare. That is only correct if compare IS subset comparison.
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

// TestCompareIsAStrictPartialOrder checks that compare really is a partial order, which is what
// entitles Dominates to be reflexive, antisymmetric and transitive. A relation with a cycle would
// make "has the stream caught up" incoherent.
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
// The canonical form
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
		// the corpus must share it. That is the property VisibleSet depends on.
		for _, other := range snapshots {
			if s.Equal(other) {
				require.Equal(t, c, canonicalize(other),
					"Equal snapshots %s and %s have different normal forms", s, other)
			}
		}
	}
}

// TestCanonicalizeCollapsesEqualButNotIdenticalSnapshots is the hazard that would sink any
// projection computed over the raw triple. These pairs are Equal under compare and have
// different (xmin, xmax, xipList), so projecting the raw triple would ship two different
// descriptions of one visible set and a remote caller would call them concurrent.
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
		})
	}
}

// TestCanonicalizeCleansIllFormedLists pins the slow path: an in-progress list that is unsorted,
// duplicated, or carries entries outside [xmin, xmax) must still produce a canonical result.
// Postgres does not emit such snapshots and the mutators do not create them, but ScanText
// validates nothing beyond parsing (snapshot.go:64-100).
func TestCanonicalizeCleansIllFormedLists(t *testing.T) {
	for _, s := range []pgSnapshot{
		{5, 10, []uint64{7, 6, 8}},       // unsorted
		{5, 10, []uint64{6, 6, 7}},       // duplicated
		{5, 10, []uint64{3, 6}},          // entry below xmin
		{5, 10, []uint64{6, 12}},         // entry at or above xmax
		{5, 10, []uint64{12, 3, 6, 6}},   // all of the above at once
		{0, 4, []uint64{3, 2, 1, 0, 99}}, // collapses to the empty snapshot
	} {
		t.Run(s.String(), func(t *testing.T) {
			c := canonicalize(s)
			require.True(t, isCanonical(c), "canonicalize produced a non-canonical snapshot: %s", c)
			require.Equal(t, c, canonicalize(c), "canonicalize is not idempotent")
		})
	}
}

// TestIllFormedSnapshotsBreakCompareItself records a known inconsistency in compare, so that the
// limit of what any causal answer can be trusted to mean is written down rather than discovered.
//
// An in-progress entry below xmin is ignored by txVisible (snapshot.go:334) but still consulted
// by otherHasMoreInfo (snapshot.go:185), so two snapshots that see exactly the same transactions
// are reported as ordered. This is unreachable from Postgres and from the mutators, which is
// asserted separately by TestMutatorsPreserveWellFormedness, but it is reachable from ScanText.
func TestIllFormedSnapshotsBreakCompareItself(t *testing.T) {
	ill := snap(5, 10, 3) // entry 3 is below xmin, so txVisible ignores it
	ok := snap(0, 10)     // no in-progress entries at all

	require.False(t, wellFormed(ill))
	require.True(t, wellFormed(ok))

	// They see exactly the same transactions.
	require.Equal(t, bruteForceVisibleSet(t, ill), bruteForceVisibleSet(t, ok))

	// And yet compare orders them, because otherHasMoreInfo consults the raw xipList
	// (snapshot.go:185) rather than asking whether this snapshot can see the entry.
	require.Equal(t, lt, ill.compare(ok),
		"expected the known inconsistency; if this now reports equal, compare was fixed")

	// VisibleSet is computed from the canonical form, so the projections agree even though
	// compare does not. A remote caller gets the answer the visible sets justify.
	require.Equal(t, canonicalize(ill), canonicalize(ok))
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
