package postgres

import (
	"slices"

	"github.com/authzed/spicedb/pkg/datastore"
)

// This file answers the causal question about Postgres revisions: did this one see everything
// that one saw.
//
// Postgres revisions are transaction snapshots, which are only PARTIALLY ordered. Two of them can
// each make visible a transaction the other cannot, and then neither happens after the other.
// Equal, LessThan and GreaterThan report that case only by omission - all three are false - which
// is why reading !LessThan as "at or after" is wrong, and wrong silently. datastore.
// VisibilityRevision exists to answer it directly instead.

// Dominates implements datastore.VisibilityRevision. It reports whether every transaction visible
// in the other revision's snapshot is also visible in this one.
//
// Postgres revisions are transaction snapshots, so this is set containment and not a comparison:
// visible(other) must be a subset of visible(pr). Two snapshots can each see a transaction the
// other cannot, and then neither dominates the other. That is the case the whole interface exists
// for, and it is why this cannot be written as !LessThan.
//
// The implementation is the one already behind pgSnapshot.compare, read in one direction.
// otherHasMoreInfo(o) reports that o knows the disposition of a transaction the receiver does not,
// which is exactly "o sees something we do not", so its negation is dominance. Routing through
// compare rather than reimplementing the subset test keeps this answer and Equal/LessThan/
// GreaterThan from ever drifting apart; TestDominanceAgreesWithCompare pins that they cannot, and
// TestCompareIsSubsetComparisonOfVisibleSets pins that compare really is subset comparison.
func (pr postgresRevision) Dominates(otherRaw datastore.Revision) bool {
	other, ok := otherRaw.(postgresRevision)
	if !ok {
		return false
	}

	switch pr.snapshot.compare(other.snapshot) {
	case equal, gt:
		return true
	case lt, concurrent:
		return false
	default:
		// compare returns one of the four above. A new result would be a bug, and claiming
		// dominance on an answer we do not understand is the dangerous direction to be wrong in.
		return false
	}
}

// VisibleSet implements datastore.VisibilityRevision, projecting the snapshot into the form a
// client in another language can compare for itself.
//
// The projection is taken over the CANONICAL snapshot, not the raw one, and that is required
// rather than tidy. snap(1,1) and snap(1,5,1,2,3,4) are Equal and have completely different
// triples; projecting the raw triple would ship two different descriptions of one visible set and
// a remote caller would report them as concurrent while a local one reports them as equal.
// canonicalize also drops in-progress entries outside [xmin, xmax), which txVisible ignores but a
// naive remote implementation would not.
//
// TestVisibleSetMatchesDominates pins that comparing two projections agrees with compare on every
// pair in the corpus, so the wire form cannot drift from the datastore's own answer.
func (pr postgresRevision) VisibleSet() datastore.VisibleSet {
	canonical := canonicalize(pr.snapshot)

	// canonicalize may alias the receiver's xipList, and VisibleSet is handed to callers who may
	// keep it, so copy rather than share. The list is one entry per in-flight write.
	return datastore.VisibleSet{
		Floor:      canonical.xmin,
		Ceiling:    canonical.xmax,
		Exceptions: slices.Clone(canonical.xipList),
	}
}

var _ datastore.VisibilityRevision = postgresRevision{}

// canonicalize returns the unique representative of the set of snapshots sharing a visible set,
// so that a projection taken over the result cannot disagree with Equal.
//
// This is required, not cosmetic. snap(1,1) and snap(1,5,1,2,3,4) are Equal under compare and
// have completely different triples; projecting the raw triple would describe one visible set two
// different ways, and two clients comparing those descriptions would call an Equal pair
// concurrent. TestCanonicalizeCollapsesEqualButNotIdenticalSnapshots pins the hazard.
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
