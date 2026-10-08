package postgres

import (
	"slices"

	"github.com/authzed/spicedb/pkg/datastore"
)

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
// GreaterThan from ever drifting apart; TestDominanceAgreesWithCompare pins that they cannot.
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
