package postgres

import (
	"testing"

	"github.com/stretchr/testify/require"

	"github.com/authzed/spicedb/pkg/datastore"
)

// revFor wraps a snapshot in a postgresRevision so the datastore.Revision methods can be reached.
func revFor(s pgSnapshot) postgresRevision {
	return postgresRevision{snapshot: s}
}

// TestDominanceAgreesWithCompare is the load-bearing test. Dominance must be exactly "equal or
// greater" under pgSnapshot.compare, for every pair in the corpus, so that Dominates can never
// drift away from Equal/LessThan/GreaterThan.
//
// It walks ordered pairs rather than unordered ones, so each pair is checked in both directions.
func TestDominanceAgreesWithCompare(t *testing.T) {
	snapshots := allTestSnapshots()

	var dominating, concurrentPairs int
	for _, a := range snapshots {
		for _, b := range snapshots {
			cmp := a.compare(b)
			want := cmp == equal || cmp == gt

			got := revFor(a).Dominates(revFor(b))
			require.Equal(t, want, got,
				"Dominates disagreed with compare=%s for a=%s b=%s", cmp, a, b)

			if got {
				dominating++
			}
			if cmp == concurrent {
				concurrentPairs++
			}
		}
	}

	// Guard against the corpus degenerating into something that cannot exercise the interesting
	// branch. If no pair is ever concurrent, this test proves far less than it appears to.
	require.Positive(t, concurrentPairs, "corpus contains no concurrent pairs, so dominance is untested where it matters")
	require.Positive(t, dominating, "corpus contains no dominating pairs")
	t.Logf("%d snapshots, %d dominating ordered pairs, %d concurrent ordered pairs",
		len(snapshots), dominating, concurrentPairs)
}

// TestDominanceIsNotNegatedLessThan pins the single most likely way for a caller to get this
// wrong. For a concurrent pair LessThan is false and Dominates is ALSO false, so !LessThan and
// Dominates are different functions, and substituting one for the other silently reports that a
// revision has caught up when it has not.
func TestDominanceIsNotNegatedLessThan(t *testing.T) {
	snapshots := allTestSnapshots()

	var divergences int
	for _, a := range snapshots {
		for _, b := range snapshots {
			ra, rb := revFor(a), revFor(b)
			if !ra.LessThan(rb) != ra.Dominates(rb) {
				divergences++
				require.Equal(t, concurrent, a.compare(b),
					"!LessThan and Dominates may only diverge on a concurrent pair, got a=%s b=%s", a, b)
			}
		}
	}

	require.Positive(t, divergences,
		"!LessThan never diverged from Dominates, so this test is not exercising the hazard it exists for")
	t.Logf("!LessThan diverges from Dominates on %d ordered pairs, every one of them concurrent", divergences)
}

// TestDominanceIsReflexiveAndAntisymmetric checks the structural properties any dominance relation
// must have. Mutual dominance must mean Equal, never anything weaker.
func TestDominanceIsReflexiveAndAntisymmetric(t *testing.T) {
	snapshots := allTestSnapshots()

	for _, a := range snapshots {
		require.True(t, revFor(a).Dominates(revFor(a)), "a revision must dominate itself: %s", a)
	}

	for i, a := range snapshots {
		for _, b := range snapshots[i:] {
			ra, rb := revFor(a), revFor(b)
			if ra.Dominates(rb) && rb.Dominates(ra) {
				require.True(t, ra.Equal(rb),
					"mutual dominance must imply Equal, got a=%s b=%s", a, b)
			}
		}
	}
}

// TestDominanceIsTransitive checks that dominance composes. A relation that is not transitive
// cannot be used as a watermark, because catching up to an intermediate revision would not imply
// catching up to an earlier one.
func TestDominanceIsTransitive(t *testing.T) {
	// Transitivity is cubic, so use the small exhaustive set rather than the full corpus.
	snapshots := exhaustiveSmallSnapshots(5)

	for _, a := range snapshots {
		for _, b := range snapshots {
			if !revFor(a).Dominates(revFor(b)) {
				continue
			}
			for _, c := range snapshots {
				if !revFor(b).Dominates(revFor(c)) {
					continue
				}
				require.True(t, revFor(a).Dominates(revFor(c)),
					"dominance is not transitive: a=%s dominates b=%s dominates c=%s", a, b, c)
			}
		}
	}
}

// TestDominanceMatchesVisibleSets proves the semantic claim in the doc comment - that dominance is
// containment of visible transaction sets - against txVisible directly, rather than against
// compare. compare is the implementation; txVisible is the meaning. Checking against the
// implementation alone could not catch the two being wrong together.
func TestDominanceMatchesVisibleSets(t *testing.T) {
	snapshots := exhaustiveSmallSnapshots(6)

	// Any txid outside [0, probeLimit) is either visible to both (below both xmins) or invisible
	// to both (at or above both xmaxes), so probing the range is exhaustive for these snapshots.
	const probeLimit = 8

	for _, a := range snapshots {
		for _, b := range snapshots {
			// visible(b) subset of visible(a): no txid is visible at b but not at a.
			wantDominates := true
			for txid := uint64(0); txid < probeLimit; txid++ {
				if b.txVisible(txid) && !a.txVisible(txid) {
					wantDominates = false
					break
				}
			}

			require.Equal(t, wantDominates, revFor(a).Dominates(revFor(b)),
				"dominance must be containment of visible sets, a=%s b=%s", a, b)
		}
	}
}

// TestDominanceOnTheConcurrentLattice is the worked example, small enough to read. It is the same
// lattice the cross-language proof uses, so a failure here and a failure there have one cause.
//
//	       all-settled  [1:5:[]]      sees {1,2,3,4}
//	        /        \
//	sees-3 [1:5:[2]]  sees-2 [1:5:[3]]   {1,3,4} vs {1,2,4}   <- CONCURRENT
//	        \        /
//	    two-in-flight [1:5:[2 3]]      sees {1,4}
func TestDominanceOnTheConcurrentLattice(t *testing.T) {
	allSettled := snap(1, 5)
	sees3 := snap(1, 5, 2)
	sees2 := snap(1, 5, 3)
	twoInFlight := snap(1, 5, 2, 3)

	// The top of the lattice dominates everything.
	require.True(t, revFor(allSettled).Dominates(revFor(sees3)))
	require.True(t, revFor(allSettled).Dominates(revFor(sees2)))
	require.True(t, revFor(allSettled).Dominates(revFor(twoInFlight)))

	// The bottom dominates nothing above it.
	require.False(t, revFor(twoInFlight).Dominates(revFor(sees3)))
	require.False(t, revFor(twoInFlight).Dominates(revFor(allSettled)))

	// Everything dominates the bottom.
	require.True(t, revFor(sees3).Dominates(revFor(twoInFlight)))
	require.True(t, revFor(sees2).Dominates(revFor(twoInFlight)))

	// The two arms are concurrent: neither dominates the other, in either direction.
	require.False(t, revFor(sees3).Dominates(revFor(sees2)), "concurrent pair must not dominate")
	require.False(t, revFor(sees2).Dominates(revFor(sees3)), "concurrent pair must not dominate")

	// And the datastore agrees that they are concurrent rather than merely equal.
	require.Equal(t, concurrent, sees3.compare(sees2))
	require.False(t, revFor(sees3).Equal(revFor(sees2)))
}

// TestVisibleSetMatchesDominates is the test the whole cross-language story rests on. The wire
// projection is compared with the generic datastore.VisibleSet.Dominates rule - the one a Python
// client reimplements - and must give the same answer as Postgres's own compare for every pair in
// the corpus.
//
// If this passes, a remote caller evaluating the projection reaches exactly the conclusion the
// datastore would. If it ever fails, the remote caller is silently wrong and nothing else catches
// it.
func TestVisibleSetMatchesDominates(t *testing.T) {
	snapshots := allTestSnapshots()

	projections := make([]datastore.VisibleSet, len(snapshots))
	for i, s := range snapshots {
		projections[i] = revFor(s).VisibleSet()
	}

	for i, a := range snapshots {
		for j, b := range snapshots {
			want := revFor(a).Dominates(revFor(b))
			got := projections[i].Dominates(projections[j])
			require.Equal(t, want, got,
				"wire projection disagreed with the datastore: a=%s %+v b=%s %+v",
				a, projections[i], b, projections[j])
		}
	}
}

// TestVisibleSetContainsMatchesTxVisible pins the projection's membership test against the
// datastore's own, which is what makes the range-with-holes encoding faithful rather than merely
// plausible.
//
// It probes the boundaries rather than a dense range. The corpus deliberately contains a snapshot
// at xmax = 1<<63, so iterating [0, xmax) is not merely slow, it does not terminate - the first
// version of this test hung and was killed by its timeout. Membership can only change at a floor,
// a ceiling or an exception, so the points either side of each are exhaustive for the property.
func TestVisibleSetContainsMatchesTxVisible(t *testing.T) {
	snapshots := allTestSnapshots()

	for _, s := range snapshots {
		projection := revFor(s).VisibleSet()

		boundaries := []uint64{0, s.xmin, s.xmax, projection.Floor, projection.Ceiling}
		boundaries = append(boundaries, s.xipList...)
		boundaries = append(boundaries, projection.Exceptions...)

		for _, b := range boundaries {
			for _, probe := range []uint64{b, b + 1, b + 2} {
				require.Equal(t, s.txVisible(probe), projection.Contains(probe),
					"projection disagreed about txid %d for snapshot %s %+v", probe, s, projection)
			}
			if b > 0 {
				require.Equal(t, s.txVisible(b-1), projection.Contains(b-1),
					"projection disagreed about txid %d for snapshot %s %+v", b-1, s, projection)
			}
		}
	}
}

// TestVisibleSetContainsMatchesTxVisibleExhaustively does the dense sweep the boundary test cannot
// afford, over the small snapshots where it is cheap. Between the two, membership is checked both
// everywhere-that-is-small and at-every-edge-that-is-large.
func TestVisibleSetContainsMatchesTxVisibleExhaustively(t *testing.T) {
	for _, s := range exhaustiveSmallSnapshots(7) {
		projection := revFor(s).VisibleSet()
		for txid := uint64(0); txid < 12; txid++ {
			require.Equal(t, s.txVisible(txid), projection.Contains(txid),
				"projection disagreed about txid %d for snapshot %s %+v", txid, s, projection)
		}
	}
}

// TestVisibleSetIsIdenticalForEqualSnapshots pins the property that makes remote equality work.
// Two Equal snapshots with different internal triples must project identically, or a client will
// call them concurrent while the server calls them equal.
func TestVisibleSetIsIdenticalForEqualSnapshots(t *testing.T) {
	// The sharp case, taken from the existing TestCompare: these are Equal and share no part of
	// their representation.
	require.Equal(t, equal, snap(1, 1).compare(snap(1, 5, 1, 2, 3, 4)))
	require.Equal(t, revFor(snap(1, 1)).VisibleSet(), revFor(snap(1, 5, 1, 2, 3, 4)).VisibleSet())

	snapshots := allTestSnapshots()
	var equalPairs int
	for i, a := range snapshots {
		for _, b := range snapshots[i:] {
			if a.compare(b) != equal {
				continue
			}
			equalPairs++
			require.Equal(t, revFor(a).VisibleSet(), revFor(b).VisibleSet(),
				"Equal snapshots must project identically: a=%s b=%s", a, b)
		}
	}
	require.Positive(t, equalPairs, "corpus contains no Equal pairs, so this proves nothing")
	t.Logf("%d Equal pairs all project identically", equalPairs)
}

// TestVisibleSetDoesNotAliasTheSnapshot pins that a caller holding a projection cannot mutate the
// revision it came from. canonicalize is documented as possibly aliasing its input.
func TestVisibleSetDoesNotAliasTheSnapshot(t *testing.T) {
	s := snap(1, 5, 2, 3)
	rev := revFor(s)

	projection := rev.VisibleSet()
	require.Equal(t, []uint64{2, 3}, projection.Exceptions)

	projection.Exceptions[0] = 99
	require.Equal(t, []uint64{2, 3}, rev.snapshot.xipList, "mutating the projection must not reach the snapshot")
	require.Equal(t, []uint64{2, 3}, rev.VisibleSet().Exceptions, "and a fresh projection must be unaffected")
}

// TestVisibleSetSizeIsOnePerInFlightWrite measures what the projection costs on the wire, since it
// is the only unbounded part of the token and the reason client-side evaluation needed justifying.
func TestVisibleSetSizeIsOnePerInFlightWrite(t *testing.T) {
	for _, inFlight := range []int{0, 1, 8, 64, 500} {
		xips := make([]uint64, 0, inFlight)
		for i := range inFlight {
			// Leave a settled transaction between each in-flight one, so canonicalize cannot
			// collapse the run and the worst case is really measured.
			xips = append(xips, uint64(1+2*i))
		}
		s := pgSnapshot{xmin: 0, xmax: uint64(2*inFlight + 2), xipList: xips}

		projection := revFor(s).VisibleSet()
		require.Len(t, projection.Exceptions, inFlight,
			"the projection carries exactly one exception per in-flight write")
	}
}

// TestDominanceRejectsForeignRevisionType pins that a revision of another concrete type never
// reports as dominated, matching what Equal, LessThan and GreaterThan do.
func TestDominanceRejectsForeignRevisionType(t *testing.T) {
	require.False(t, revFor(snap(1, 5)).Dominates(datastore.NoRevision))
}

// TestPostgresRevisionImplementsVisibilityRevision pins the capability itself. A type assertion is
// how a consumer discovers it, so losing the method would silently downgrade every caller to the
// totally-ordered fallback - which is wrong for Postgres and wrong without erroring.
func TestPostgresRevisionImplementsVisibilityRevision(t *testing.T) {
	var rev datastore.Revision = revFor(snap(1, 5))

	vr, ok := rev.(datastore.VisibilityRevision)
	require.True(t, ok, "postgresRevision must implement datastore.VisibilityRevision")
	require.True(t, vr.Dominates(rev))
}
