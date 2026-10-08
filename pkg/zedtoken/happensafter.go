package zedtoken

import (
	"errors"
	"fmt"

	v1 "github.com/authzed/authzed-go/proto/authzed/api/v1"

	"github.com/authzed/spicedb/pkg/datastore"
)

// errCausalityError wraps failures of the causal comparison. It is deliberately not
// errDecodeError: a token can decode perfectly and still fail here, and saying "error decoding"
// would be false.
const errCausalityError = "error comparing zedtokens: %w"

// ErrForeignToken is returned when a zedtoken carries a datastore unique ID that does not match
// the datastore it is being compared against. Causality is only defined within a single datastore,
// so a foreign token has no relationship to a local one and refusing is the only safe answer.
//
// NOTE FOR INTEGRATION: the sort-key work on branch feat/zedtoken-sortkey declares
// ErrDatastoreIDMismatch for this identical condition. The two branches were written in parallel
// against a base that had neither. Whoever merges them should keep one sentinel and delete the
// other; nothing but the name differs.
var ErrForeignToken = errors.New("zedtoken was issued by a different datastore")

// Ordering is the causal relationship between two zedtokens. It has four values because three of
// the four answers are not "yes", and because the fourth - OrderingConcurrent - is a real answer
// rather than a failure to compute one.
//
// A revision names the set of writes visible at it. For a totally ordered datastore that set only
// grows, so every pair is Before, After or Equal. For a partially ordered one - PostgreSQL, whose
// revisions are transaction snapshots - two revisions can each see a write the other cannot, and
// the honest answer is Concurrent.
type Ordering int

const (
	// OrderingBefore means a happens before b: everything visible at a is visible at b, and b
	// sees at least one write a does not.
	OrderingBefore Ordering = iota

	// OrderingAfter means a happens after b: everything visible at b is visible at a, and a sees
	// at least one write b does not.
	OrderingAfter

	// OrderingEqual means a and b make exactly the same writes visible. The tokens need not be
	// byte-identical, and on PostgreSQL two Equal snapshots routinely have different internal
	// representations.
	OrderingEqual

	// OrderingConcurrent means neither token happens after the other: each makes visible at least
	// one write the other does not.
	//
	// ⚠️ This is an answer, not an error. It cannot be collapsed into Before or After without
	// inventing a relationship that does not exist. A caller that must pick a winner anyway
	// should sort by sort key, which imposes a stable arbitrary order on exactly these pairs, and
	// must not then describe the result as causal.
	//
	// Only partially ordered datastores can produce it. PostgreSQL is the only one SpiceDB ships.
	OrderingConcurrent
)

func (o Ordering) String() string {
	switch o {
	case OrderingBefore:
		return "BEFORE"
	case OrderingAfter:
		return "AFTER"
	case OrderingEqual:
		return "EQUAL"
	case OrderingConcurrent:
		return "CONCURRENT"
	default:
		return fmt.Sprintf("Ordering(%d)", int(o))
	}
}

// CompareCausality reports the causal relationship between two zedtokens: whether a happens
// before b, after it, at the same point, or concurrently with it.
//
// This is what answers "is this zedtoken after that one". It is NOT the sort-key comparison, and
// the two differ on exactly the pairs that matter:
//
//   - A sort key orders every pair, concurrent ones included, by picking a stable but arbitrary
//     winner. It is for putting tokens in a list.
//   - CompareCausality refuses to pick, and returns OrderingConcurrent instead. It is for
//     deciding whether one revision actually saw what another saw.
//
// So a token can sort after another while being concurrent with it. Using sort order to answer a
// causal question is wrong, and wrong silently.
//
// Errors are those of DecodeRevision, plus ErrForeignToken when the tokens name different
// datastores. Legacy tokens carrying no datastore unique ID are accepted on the same grounds the
// rest of this package accepts them: their revision is already trusted enough to read at.
func CompareCausality(a, b *v1.ZedToken, ds RevisionHolder) (Ordering, error) {
	aRev, err := comparableRevision(a, ds)
	if err != nil {
		return OrderingConcurrent, err
	}

	bRev, err := comparableRevision(b, ds)
	if err != nil {
		return OrderingConcurrent, err
	}

	aDominatesB := dominates(aRev, bRev)
	bDominatesA := dominates(bRev, aRev)

	switch {
	case aDominatesB && bDominatesA:
		return OrderingEqual, nil
	case aDominatesB:
		return OrderingAfter, nil
	case bDominatesA:
		return OrderingBefore, nil
	default:
		return OrderingConcurrent, nil
	}
}

// Reached reports whether observed is at or after target: whether every write visible at target is
// also visible at observed.
//
// This is the watermark primitive, and it is what answers "wait until at least this zedtoken has
// come through". Given a stream of zedtokens - from the Watch API, say - the first one for which
// Reached returns true is the point at which the stream has caught up to target.
//
//	for token := range stream {
//	    caughtUp, err := zedtoken.Reached(token, target, ds)
//	    if err != nil { return err }
//	    if caughtUp { break }
//	}
//
// Unlike CompareCausality this is a yes-or-no question with an exact answer for every pair,
// including concurrent ones. ⚠️ The answer for a concurrent pair is NO: a revision that is
// concurrent with target is missing a write target has, so it has not reached it, however it
// happens to sort. It is exactly equivalent to CompareCausality returning OrderingAfter or
// OrderingEqual, and is offered separately because the bool is the shape callers want and because
// writing it by hand invites the mistake of also accepting OrderingConcurrent.
//
// Errors are those of CompareCausality.
func Reached(observed, target *v1.ZedToken, ds RevisionHolder) (bool, error) {
	observedRev, err := comparableRevision(observed, ds)
	if err != nil {
		return false, err
	}

	targetRev, err := comparableRevision(target, ds)
	if err != nil {
		return false, err
	}

	return dominates(observedRev, targetRev), nil
}

// dominates reports whether every write visible at b is also visible at a.
//
// A revision type that implements datastore.VisibilityRevision answers for itself, because only it
// knows how its visible set is shaped. A type that declines the interface is asserting that it is
// totally ordered, and for a total order dominance is exactly "at or after" - so Equal or
// GreaterThan is the complete answer, and OrderingConcurrent can never arise. Every revision type
// SpiceDB ships except PostgreSQL takes that path.
//
// ⚠️ The fallback is deliberately Equal || GreaterThan and NOT !LessThan. They agree for a total
// order and disagree for a partial one, where both LessThan and GreaterThan are false. Writing it
// as !LessThan would make a future partially ordered type that forgot to implement the interface
// silently report dominance for every concurrent pair, which is the worst available failure: a
// confident wrong answer. Written this way it under-reports instead, which a test can catch.
func dominates(a, b datastore.Revision) bool {
	if vr, ok := a.(datastore.VisibilityRevision); ok {
		return vr.Dominates(b)
	}

	return a.Equal(b) || a.GreaterThan(b)
}

// comparableRevision decodes a token and returns its revision, refusing tokens that belong to a
// different datastore.
//
// It goes through DecodeRevision so the revision parsing and the datastore-ID check cannot drift
// from the ones every other caller sees. The schema hash is discarded: it has no bearing on
// causality, and two tokens naming one revision with different schema hashes are OrderingEqual.
func comparableRevision(encoded *v1.ZedToken, ds RevisionHolder) (datastore.Revision, error) {
	decoded, err := DecodeRevision(encoded, ds)
	if err != nil {
		return nil, err
	}

	switch decoded.Status {
	case StatusValid:
		// Provenance proven: the token names this datastore.

	case StatusLegacyEmptyDatastoreID:
		// Provenance unprovable but accepted; see CompareCausality's doc comment.

	case StatusMismatchedDatastoreID:
		return nil, fmt.Errorf(errCausalityError, ErrForeignToken)

	default:
		return nil, fmt.Errorf(errCausalityError, fmt.Errorf("unexpected zedtoken status: %d", decoded.Status))
	}

	return decoded.Revision, nil
}
