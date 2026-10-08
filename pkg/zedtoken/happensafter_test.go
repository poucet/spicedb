package zedtoken

import (
	"context"
	"errors"
	"fmt"
	"slices"
	"strconv"
	"strings"
	"testing"

	"github.com/stretchr/testify/require"

	v1 "github.com/authzed/authzed-go/proto/authzed/api/v1"

	"github.com/authzed/spicedb/internal/datastore/revisions"
	"github.com/authzed/spicedb/pkg/datalayer"
	"github.com/authzed/spicedb/pkg/datastore"
)

const causalityTestDatastoreID = "f0e1d2c3-b4a5-6978-8a9b-0c1d2e3f4051"

// setRevision is a revision whose visible set is written out explicitly, so a test can state a
// partial order directly instead of deriving it from a datastore's internals.
//
// It stands in for a PostgreSQL transaction snapshot without importing the postgres package: the
// property under test is that pkg/zedtoken handles a PARTIALLY ORDERED revision type correctly,
// and a type whose partial order is visible in the test source demonstrates that better than one
// whose order has to be taken on trust. The real Postgres implementation is proved against its
// own snapshot corpus in internal/datastore/postgres/dominance_test.go, and the two are proved
// together end to end by the cross-language harness.
//
// The string form is a sorted comma-separated list, so revisions round trip through a zedtoken.
type setRevision struct {
	visible []uint64
}

func newSetRevision(visible ...uint64) setRevision {
	sorted := slices.Clone(visible)
	slices.Sort(sorted)
	return setRevision{visible: sorted}
}

func (sr setRevision) String() string {
	parts := make([]string, 0, len(sr.visible))
	for _, v := range sr.visible {
		parts = append(parts, strconv.FormatUint(v, 10))
	}
	return strings.Join(parts, ",")
}

func (sr setRevision) contains(v uint64) bool {
	_, found := slices.BinarySearch(sr.visible, v)
	return found
}

// Dominates implements datastore.VisibilityRevision: every element of other must be present here.
func (sr setRevision) Dominates(otherRaw datastore.Revision) bool {
	other, ok := otherRaw.(setRevision)
	if !ok {
		return false
	}

	for _, v := range other.visible {
		if !sr.contains(v) {
			return false
		}
	}
	return true
}

func (sr setRevision) Equal(rhs datastore.Revision) bool {
	other, ok := rhs.(setRevision)
	return ok && slices.Equal(sr.visible, other.visible)
}

// LessThan and GreaterThan are strict subset and strict superset, so a pair of sets that each
// contain something the other does not reports false from all three - exactly how a concurrent
// Postgres snapshot pair behaves.
func (sr setRevision) LessThan(rhs datastore.Revision) bool {
	other, ok := rhs.(setRevision)
	return ok && !sr.Equal(other) && other.Dominates(sr)
}

func (sr setRevision) GreaterThan(rhs datastore.Revision) bool {
	other, ok := rhs.(setRevision)
	return ok && !sr.Equal(other) && sr.Dominates(other)
}

func (sr setRevision) ByteSortable() bool { return true }

var (
	_ datastore.Revision           = setRevision{}
	_ datastore.VisibilityRevision = setRevision{}
)

// setHolder is a RevisionHolder over setRevision.
type setHolder struct{ datastoreUniqueID string }

func (sh setHolder) UniqueID(_ context.Context) (string, error) { return sh.datastoreUniqueID, nil }

func (sh setHolder) RevisionFromString(s string) (datastore.Revision, error) {
	if s == "" {
		return newSetRevision(), nil
	}

	var visible []uint64
	for _, part := range strings.Split(s, ",") {
		parsed, err := strconv.ParseUint(part, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("bad set revision %q: %w", s, err)
		}
		visible = append(visible, parsed)
	}
	return newSetRevision(visible...), nil
}

var _ RevisionHolder = setHolder{}

// The lattice every causality test is built on. It mirrors the Postgres fixture used by the
// cross-language proof, so a failure in either has one cause.
//
//	     allSettled {1,2,3,4}
//	      /        \
//	sees3 {1,3,4}  sees2 {1,2,4}     <- CONCURRENT with each other
//	      \        /
//	  twoInFlight {1,4}
func latticeRevisions() (allSettled, sees3, sees2, twoInFlight setRevision) {
	return newSetRevision(1, 2, 3, 4),
		newSetRevision(1, 3, 4),
		newSetRevision(1, 2, 4),
		newSetRevision(1, 4)
}

func mustToken(t *testing.T, rev datastore.Revision, holder setHolder) *v1.ZedToken {
	t.Helper()
	token, err := NewFromRevision(context.Background(), rev, datalayer.NoSchemaHashInLegacyZedToken, holder)
	require.NoError(t, err)
	return token
}

func TestCompareCausalityOnTheLattice(t *testing.T) {
	holder := setHolder{causalityTestDatastoreID}
	allSettled, sees3, sees2, twoInFlight := latticeRevisions()

	tc := []struct {
		name string
		a, b datastore.Revision
		want Ordering
	}{
		{"top after left arm", allSettled, sees3, OrderingAfter},
		{"top after right arm", allSettled, sees2, OrderingAfter},
		{"top after bottom", allSettled, twoInFlight, OrderingAfter},
		{"bottom before top", twoInFlight, allSettled, OrderingBefore},
		{"bottom before left arm", twoInFlight, sees3, OrderingBefore},
		{"left arm after bottom", sees3, twoInFlight, OrderingAfter},
		{"a revision equals itself", sees3, newSetRevision(1, 3, 4), OrderingEqual},
		{"the two arms are concurrent", sees3, sees2, OrderingConcurrent},
		{"concurrency is symmetric", sees2, sees3, OrderingConcurrent},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			got, err := CompareCausality(mustToken(t, tt.a, holder), mustToken(t, tt.b, holder), holder)
			require.NoError(t, err)
			require.Equal(t, tt.want, got, "want %s, got %s", tt.want, got)
		})
	}
}

// TestReachedIsFalseForConcurrent is the assertion the watermark use case rests on. A revision
// that is concurrent with the target has NOT reached it: it is missing a write the target has,
// however the two happen to sort.
func TestReachedIsFalseForConcurrent(t *testing.T) {
	holder := setHolder{causalityTestDatastoreID}
	allSettled, sees3, sees2, twoInFlight := latticeRevisions()

	tc := []struct {
		name             string
		observed, target datastore.Revision
		want             bool
	}{
		{"reached a strictly earlier revision", allSettled, twoInFlight, true},
		{"reached an equal revision", sees3, newSetRevision(1, 3, 4), true},
		{"reached itself", sees3, sees3, true},
		{"has not reached a later revision", twoInFlight, allSettled, false},
		{"has NOT reached a concurrent revision", sees3, sees2, false},
		{"nor the other way round", sees2, sees3, false},
	}

	for _, tt := range tc {
		t.Run(tt.name, func(t *testing.T) {
			got, err := Reached(mustToken(t, tt.observed, holder), mustToken(t, tt.target, holder), holder)
			require.NoError(t, err)
			require.Equal(t, tt.want, got)
		})
	}
}

// TestReachedIsNotNegatedLessThan pins the hazard at the public API boundary. A caller reaching
// for "not before, therefore caught up" gets the wrong answer on a concurrent pair, and gets it
// without an error.
func TestReachedIsNotNegatedLessThan(t *testing.T) {
	_, sees3, sees2, _ := latticeRevisions()

	// The tempting shortcut says sees3 has caught up to sees2, because it is not before it.
	require.False(t, sees3.LessThan(sees2), "the shortcut's premise: neither is less than the other")
	require.False(t, sees3.GreaterThan(sees2))
	require.False(t, sees3.Equal(sees2))

	// The truth is that it has not: sees2 can see transaction 2, and sees3 cannot.
	require.False(t, sees3.Dominates(sees2), "but it has NOT caught up")
	require.True(t, sees2.contains(2))
	require.False(t, sees3.contains(2))
}

// TestReachedAgreesWithCompareCausality pins that the bool and the four-valued answer can never
// disagree, so a caller may use whichever fits without reasoning about the difference.
func TestReachedAgreesWithCompareCausality(t *testing.T) {
	holder := setHolder{causalityTestDatastoreID}
	allSettled, sees3, sees2, twoInFlight := latticeRevisions()
	all := []datastore.Revision{allSettled, sees3, sees2, twoInFlight}

	var sawConcurrent bool
	for _, a := range all {
		for _, b := range all {
			ta, tb := mustToken(t, a, holder), mustToken(t, b, holder)

			ord, err := CompareCausality(ta, tb, holder)
			require.NoError(t, err)

			reached, err := Reached(ta, tb, holder)
			require.NoError(t, err)

			require.Equal(t, ord == OrderingAfter || ord == OrderingEqual, reached,
				"Reached must be exactly AFTER or EQUAL, got ordering %s reached %v for %s vs %s", ord, reached, a, b)

			if ord == OrderingConcurrent {
				sawConcurrent = true
			}
		}
	}
	require.True(t, sawConcurrent, "the lattice must produce a concurrent pair or this proves little")
}

// TestCausalityIsATotalFunction checks that every pair gets exactly one of the four answers, and
// that Before and After are mirror images.
func TestCausalityIsATotalFunction(t *testing.T) {
	holder := setHolder{causalityTestDatastoreID}
	allSettled, sees3, sees2, twoInFlight := latticeRevisions()
	all := []datastore.Revision{allSettled, sees3, sees2, twoInFlight, newSetRevision(), newSetRevision(9)}

	for _, a := range all {
		for _, b := range all {
			ta, tb := mustToken(t, a, holder), mustToken(t, b, holder)

			forward, err := CompareCausality(ta, tb, holder)
			require.NoError(t, err)
			reverse, err := CompareCausality(tb, ta, holder)
			require.NoError(t, err)

			var want Ordering
			switch forward {
			case OrderingAfter:
				want = OrderingBefore
			case OrderingBefore:
				want = OrderingAfter
			case OrderingEqual, OrderingConcurrent:
				want = forward
			}
			require.Equal(t, want, reverse, "asymmetry for %s vs %s: %s then %s", a, b, forward, reverse)
		}
	}
}

// TestCausalityFallbackForTotallyOrderedRevisions proves that a revision type which declines
// VisibilityRevision still works, and can never report CONCURRENT. Transaction ID revisions take
// this path, as do HLC and timestamp revisions.
func TestCausalityFallbackForTotallyOrderedRevisions(t *testing.T) {
	holder := revisions.CommonDecoder{
		Kind:              revisions.TransactionID,
		DatastoreUniqueID: causalityTestDatastoreID,
	}

	// Guard the premise: if this type ever gains the interface, the fallback stops being covered.
	var rev datastore.Revision = revisions.NewForTransactionID(5)
	_, implementsVisibility := rev.(datastore.VisibilityRevision)
	require.False(t, implementsVisibility,
		"TransactionIDRevision now implements VisibilityRevision, so this test no longer covers the fallback")

	tokenFor := func(n uint64) *v1.ZedToken {
		token, err := NewFromRevision(context.Background(), revisions.NewForTransactionID(n),
			datalayer.NoSchemaHashInLegacyZedToken, holder)
		require.NoError(t, err)
		return token
	}

	for _, tt := range []struct {
		name string
		a, b uint64
		want Ordering
	}{
		{"later is after", 10, 5, OrderingAfter},
		{"earlier is before", 5, 10, OrderingBefore},
		{"same is equal", 7, 7, OrderingEqual},
		// "9" sorts above "10" as a string; causality is not string order.
		{"nine is before ten", 9, 10, OrderingBefore},
	} {
		t.Run(tt.name, func(t *testing.T) {
			got, err := CompareCausality(tokenFor(tt.a), tokenFor(tt.b), holder)
			require.NoError(t, err)
			require.Equal(t, tt.want, got)
			require.NotEqual(t, OrderingConcurrent, got,
				"a totally ordered revision type must never report CONCURRENT")
		})
	}

	reached, err := Reached(tokenFor(10), tokenFor(5), holder)
	require.NoError(t, err)
	require.True(t, reached)

	reached, err = Reached(tokenFor(5), tokenFor(10), holder)
	require.NoError(t, err)
	require.False(t, reached)
}

// TestCausalityRejectsForeignTokens pins that tokens from another datastore are refused rather
// than compared. Causality is only defined within one datastore.
func TestCausalityRejectsForeignTokens(t *testing.T) {
	local := setHolder{causalityTestDatastoreID}
	foreign := setHolder{"99999999-aaaa-bbbb-cccc-dddddddddddd"}

	localToken := mustToken(t, newSetRevision(1, 2), local)
	foreignToken := mustToken(t, newSetRevision(1, 2, 3), foreign)

	_, err := CompareCausality(localToken, foreignToken, local)
	require.ErrorIs(t, err, ErrForeignToken)

	_, err = CompareCausality(foreignToken, localToken, local)
	require.ErrorIs(t, err, ErrForeignToken)

	_, err = Reached(foreignToken, localToken, local)
	require.ErrorIs(t, err, ErrForeignToken)
}

// TestCausalityAcceptsLegacyTokens pins the same decision the rest of the package makes: a token
// with no datastore unique ID is compared rather than refused.
func TestCausalityAcceptsLegacyTokens(t *testing.T) {
	holder := setHolder{causalityTestDatastoreID}

	earlier, err := NewFromRevisionNoDatastoreID(newSetRevision(1), datalayer.NoSchemaHashInLegacyZedToken)
	require.NoError(t, err)
	later, err := NewFromRevisionNoDatastoreID(newSetRevision(1, 2), datalayer.NoSchemaHashInLegacyZedToken)
	require.NoError(t, err)

	got, err := CompareCausality(later, earlier, holder)
	require.NoError(t, err)
	require.Equal(t, OrderingAfter, got)
}

// TestCausalityErrorCases covers the inputs that must fail rather than answer.
func TestCausalityErrorCases(t *testing.T) {
	holder := setHolder{causalityTestDatastoreID}
	good := mustToken(t, newSetRevision(1), holder)

	for _, tt := range []struct {
		name string
		a, b *v1.ZedToken
	}{
		{"nil first", nil, good},
		{"nil second", good, nil},
		{"malformed first", &v1.ZedToken{Token: "!!!not base64!!!"}, good},
		{"malformed second", good, &v1.ZedToken{Token: "!!!not base64!!!"}},
	} {
		t.Run(tt.name, func(t *testing.T) {
			_, err := CompareCausality(tt.a, tt.b, holder)
			require.Error(t, err)

			_, err = Reached(tt.a, tt.b, holder)
			require.Error(t, err)
		})
	}
}

// TestCausalityErrorsAreUnwrappable pins that the sentinel survives the fmt.Errorf wrapping, so
// callers can branch on it with errors.Is.
func TestCausalityErrorsAreUnwrappable(t *testing.T) {
	local := setHolder{causalityTestDatastoreID}
	foreignToken := mustToken(t, newSetRevision(1), setHolder{"99999999-aaaa-bbbb-cccc-dddddddddddd"})

	_, err := CompareCausality(foreignToken, foreignToken, local)
	require.Error(t, err)
	require.True(t, errors.Is(err, ErrForeignToken), "ErrForeignToken must survive wrapping")
}

// TestCausalityIgnoresSchemaHash pins that two tokens naming one revision compare EQUAL even when
// they were issued with different schema hashes. The schema hash has no bearing on causality.
func TestCausalityIgnoresSchemaHash(t *testing.T) {
	holder := setHolder{causalityTestDatastoreID}
	rev := newSetRevision(1, 2, 3)

	withHash, err := NewFromRevision(context.Background(), rev, datalayer.SchemaHash("aaaaaaaaaaaaaaaa"), holder)
	require.NoError(t, err)
	withOther, err := NewFromRevision(context.Background(), rev, datalayer.SchemaHash("bbbbbbbbbbbbbbbb"), holder)
	require.NoError(t, err)

	require.NotEqual(t, withHash.Token, withOther.Token, "the tokens must actually differ")

	got, err := CompareCausality(withHash, withOther, holder)
	require.NoError(t, err)
	require.Equal(t, OrderingEqual, got)
}

// TestOrderingString pins the names, because they appear in logs and in the cross-language proof.
func TestOrderingString(t *testing.T) {
	require.Equal(t, "BEFORE", OrderingBefore.String())
	require.Equal(t, "AFTER", OrderingAfter.String())
	require.Equal(t, "EQUAL", OrderingEqual.String())
	require.Equal(t, "CONCURRENT", OrderingConcurrent.String())
	require.Equal(t, "Ordering(42)", Ordering(42).String())
}

// TestWatermarkOverAStream is the user's second question, written the way they would write it:
// consume tokens until the stream has caught up to a target.
//
// The stream deliberately contains a revision that is CONCURRENT with the target and arrives
// before any revision that dominates it. A watermark built on ordering rather than dominance
// would stop early, at the concurrent one, and the caller would believe their write was visible
// when it was not.
func TestWatermarkOverAStream(t *testing.T) {
	holder := setHolder{causalityTestDatastoreID}
	_, sees3, sees2, twoInFlight := latticeRevisions()

	target := sees2 // the write we are waiting for: visible set {1,2,4}

	stream := []datastore.Revision{
		twoInFlight,                // {1,4}      strictly before the target
		sees3,                      // {1,3,4}    CONCURRENT with the target: the trap
		newSetRevision(1, 2, 4),    // {1,2,4}    equal to the target: this is the answer
		newSetRevision(1, 2, 3, 4), // later still
	}

	var stoppedAt int = -1
	for i, rev := range stream {
		caughtUp, err := Reached(mustToken(t, rev, holder), mustToken(t, target, holder), holder)
		require.NoError(t, err)
		if caughtUp {
			stoppedAt = i
			break
		}
	}

	require.Equal(t, 2, stoppedAt, "the watermark must stop at the first revision that dominates the target")

	// And prove the trap is real: the concurrent event at index 1 is not ordered before the
	// target either, so a watermark written as "stop when not before" would have stopped there.
	require.False(t, stream[1].LessThan(target), "the trap: the concurrent event is not before the target")
}
