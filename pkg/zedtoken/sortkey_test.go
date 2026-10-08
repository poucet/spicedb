package zedtoken

import (
	"bytes"
	"context"
	"slices"
	"strconv"
	"testing"

	"github.com/stretchr/testify/require"

	v1 "github.com/authzed/authzed-go/proto/authzed/api/v1"

	"github.com/authzed/spicedb/internal/datastore/revisions"
	"github.com/authzed/spicedb/pkg/datalayer"
	"github.com/authzed/spicedb/pkg/datastore"
)

const testDatastoreID = "6349aaf2-37cd-47b9-84e8-fe5fa6e2dead"

// unsortableRevision is a datastore.Revision that does not implement datastore.SortKeyRevision.
//
// No datastore SpiceDB ships produces such a revision any more. Postgres was the last one, and it
// gained a sort key once it became clear that a key only has to avoid contradicting the partial
// order rather than reproduce it. What is left in the tree is datastore.NoRevision.
//
// The tests below are therefore defending a contract rather than describing a datastore: a
// revision type may decline the capability, and when one does, the zedtoken API must say so with
// ErrNotSortable instead of inventing an order. A local type states that contract without
// depending on some real datastore continuing to decline it.
type unsortableRevision uint64

func (ur unsortableRevision) ByteSortable() bool { return false }

func (ur unsortableRevision) Equal(rhs datastore.Revision) bool {
	other, ok := rhs.(unsortableRevision)
	return ok && ur == other
}

func (ur unsortableRevision) GreaterThan(rhs datastore.Revision) bool {
	other, ok := rhs.(unsortableRevision)
	return ok && ur > other
}

func (ur unsortableRevision) LessThan(rhs datastore.Revision) bool {
	other, ok := rhs.(unsortableRevision)
	return ok && ur < other
}

func (ur unsortableRevision) String() string {
	return strconv.FormatUint(uint64(ur), 10)
}

var _ datastore.Revision = unsortableRevision(0)

// unsortableHolder is a RevisionHolder whose revisions have no sort key.
type unsortableHolder struct {
	datastoreUniqueID string
}

func (uh unsortableHolder) UniqueID(_ context.Context) (string, error) {
	return uh.datastoreUniqueID, nil
}

func (uh unsortableHolder) RevisionFromString(s string) (datastore.Revision, error) {
	parsed, err := revisions.CommonDecoder{Kind: revisions.TransactionID}.RevisionFromString(s)
	if err != nil {
		return nil, err
	}
	return unsortableRevision(uint64(parsed.(revisions.TransactionIDRevision))), nil
}

var _ RevisionHolder = unsortableHolder{}

func decoderFor(kind revisions.RevisionKind) revisions.CommonDecoder {
	return revisions.CommonDecoder{Kind: kind, DatastoreUniqueID: testDatastoreID}
}

func mustToken(t *testing.T, rev datastore.Revision) *v1.ZedToken {
	t.Helper()
	encoded, err := newFromRevision(rev, testDatastoreID, datalayer.NoSchemaHashInLegacyZedToken)
	require.NoError(t, err)
	return encoded
}

func mustHLCRev(t *testing.T, s string) datastore.Revision {
	t.Helper()
	rev, err := revisions.HLCRevisionFromString(s)
	require.NoError(t, err)
	return rev
}

// sortKeyRevisionSets are sample revisions, one set per sortable revision type. The sets are
// deliberately not in order: every test derives the expected order from LessThan, so that the
// keys are checked against the revision type's own comparison rather than against a hand-written
// list that could be wrong in the same way the key is.
func sortKeyRevisionSets(t *testing.T) []struct {
	name string
	kind revisions.RevisionKind
	revs []datastore.Revision
} {
	t.Helper()
	return []struct {
		name string
		kind revisions.RevisionKind
		revs []datastore.Revision
	}{
		{
			name: "transaction ID",
			kind: revisions.TransactionID,
			revs: []datastore.Revision{
				revisions.NewForTransactionID(0),
				revisions.NewForTransactionID(1),
				revisions.NewForTransactionID(2),
				revisions.NewForTransactionID(9),
				// 10 sorts below 9 as a decimal string; the key must not.
				revisions.NewForTransactionID(10),
				revisions.NewForTransactionID(100),
				revisions.NewForTransactionID(1000),
				revisions.NewForTransactionID(256),
				revisions.NewForTransactionID(1621538189028928000),
			},
		},
		{
			name: "timestamp",
			kind: revisions.Timestamp,
			revs: []datastore.Revision{
				revisions.NewForTimestamp(0),
				revisions.NewForTimestamp(1),
				revisions.NewForTimestamp(9),
				revisions.NewForTimestamp(10),
				revisions.NewForTimestamp(1000),
				revisions.NewForTimestamp(1621538189028928000),
			},
		},
		{
			name: "hybrid logical clock",
			kind: revisions.HybridLogicalClock,
			revs: []datastore.Revision{
				mustHLCRev(t, "1234"),
				mustHLCRev(t, "1234.0000000001"),
				mustHLCRev(t, "1234.0000000002"),
				mustHLCRev(t, "1235"),
				mustHLCRev(t, "1693540940373045727.0000000001"),
			},
		},
	}
}

// ascendingRevisions returns the set's revisions sorted by LessThan, which is the order the sort
// keys are required to reproduce.
func ascendingRevisions(revs []datastore.Revision) []datastore.Revision {
	sorted := slices.Clone(revs)
	slices.SortStableFunc(sorted, func(a, b datastore.Revision) int {
		switch {
		case a.LessThan(b):
			return -1
		case a.GreaterThan(b):
			return 1
		default:
			return 0
		}
	})
	return sorted
}

func TestSortKeyAgreesWithRevisionOrder(t *testing.T) {
	for _, tc := range sortKeyRevisionSets(t) {
		t.Run(tc.name, func(t *testing.T) {
			ds := decoderFor(tc.kind)
			revs := ascendingRevisions(tc.revs)

			for i, a := range revs {
				for j, b := range revs {
					aKey, err := SortKey(mustToken(t, a), ds)
					require.NoError(t, err)
					bKey, err := SortKey(mustToken(t, b), ds)
					require.NoError(t, err)

					cmp := bytes.Compare(aKey, bKey)
					switch {
					case a.LessThan(b):
						require.Negative(t, cmp, "%s (%d) should key below %s (%d)", a, i, b, j)
					case a.GreaterThan(b):
						require.Positive(t, cmp, "%s (%d) should key above %s (%d)", a, i, b, j)
					default:
						require.True(t, a.Equal(b))
						require.Zero(t, cmp, "%s and %s are equal and should key equal", a, b)
					}
				}
			}
		})
	}
}

func TestSortKeyRoundTripSortsEncodedTokens(t *testing.T) {
	for _, tc := range sortKeyRevisionSets(t) {
		t.Run(tc.name, func(t *testing.T) {
			ds := decoderFor(tc.kind)
			want := ascendingRevisions(tc.revs)

			// Encode in the reverse of the expected order, so a no-op sort cannot pass.
			tokens := make([]*v1.ZedToken, 0, len(want))
			for i := len(want) - 1; i >= 0; i-- {
				tokens = append(tokens, mustToken(t, want[i]))
			}

			require.NoError(t, Sort(tokens, ds))

			for i, token := range tokens {
				decoded, err := DecodeRevision(token, ds)
				require.NoError(t, err)
				require.True(t, want[i].Equal(decoded.Revision),
					"position %d: want %s, got %s", i, want[i], decoded.Revision)
			}
		})
	}
}

// TestSortKeyIsExactlyTheRevisionSortKey pins the key to datastore.SortKeyRevision's bytes and
// nothing else: no schema hash, no datastore ID prefix, no framing of our own. Any of those would
// partition the key space ahead of the revision and destroy revision order, and all of them would
// also break the durability promise the key inherits.
func TestSortKeyIsExactlyTheRevisionSortKey(t *testing.T) {
	for _, tc := range sortKeyRevisionSets(t) {
		t.Run(tc.name, func(t *testing.T) {
			ds := decoderFor(tc.kind)

			for _, rev := range tc.revs {
				skr, ok := rev.(datastore.SortKeyRevision)
				require.True(t, ok, "%T should implement datastore.SortKeyRevision", rev)

				key, err := SortKey(mustToken(t, rev), ds)
				require.NoError(t, err)
				require.Equal(t, skr.AppendSortKey(nil), key, "revision %s", rev)
			}
		})
	}
}

func TestSortKeyIsPrefixFree(t *testing.T) {
	for _, tc := range sortKeyRevisionSets(t) {
		t.Run(tc.name, func(t *testing.T) {
			ds := decoderFor(tc.kind)

			keys := make([][]byte, 0, len(tc.revs))
			for _, rev := range tc.revs {
				key, err := SortKey(mustToken(t, rev), ds)
				require.NoError(t, err)
				require.NotEmpty(t, key)
				keys = append(keys, key)
			}

			for i, a := range keys {
				for j, b := range keys {
					if i == j {
						continue
					}
					if len(a) < len(b) {
						require.False(t, bytes.Equal(a, b[:len(a)]),
							"key %d is a prefix of key %d", i, j)
					}
				}
			}
		})
	}
}

// TestSortKeyPrefixFreedomSurvivesSuffixes is the point of prefix freedom: a sort key used as one
// field of a longer key keeps its order whatever follows it.
func TestSortKeyPrefixFreedomSurvivesSuffixes(t *testing.T) {
	ds := decoderFor(revisions.TransactionID)

	lowKey, err := SortKey(mustToken(t, revisions.NewForTransactionID(100)), ds)
	require.NoError(t, err)
	highKey, err := SortKey(mustToken(t, revisions.NewForTransactionID(1000)), ds)
	require.NoError(t, err)

	// The decimal strings "100" and "1000" are a prefix pair, so a suffix can reorder them.
	require.Negative(t, bytes.Compare(lowKey, highKey))

	lowLonger := append(slices.Clone(lowKey), 0xFF, 0xFF, 0xFF)
	highShorter := append(slices.Clone(highKey), 0x00)
	require.Negative(t, bytes.Compare(lowLonger, highShorter),
		"a suffix must not reorder two sort keys")
}

func TestAppendSortKeyPreservesDst(t *testing.T) {
	ds := decoderFor(revisions.TransactionID)
	token := mustToken(t, revisions.NewForTransactionID(42))

	bare, err := SortKey(token, ds)
	require.NoError(t, err)
	require.NotEmpty(t, bare)

	prefix := []byte("relationship/")
	extended, err := AppendSortKey(slices.Clone(prefix), token, ds)
	require.NoError(t, err)

	require.Equal(t, prefix, extended[:len(prefix)], "dst must be preserved")
	require.Equal(t, bare, extended[len(prefix):], "key must be appended after dst")
	require.Len(t, extended, len(prefix)+len(bare))
}

func TestAppendSortKeyNilDstMatchesSortKey(t *testing.T) {
	ds := decoderFor(revisions.TransactionID)
	token := mustToken(t, revisions.NewForTransactionID(7))

	appended, err := AppendSortKey(nil, token, ds)
	require.NoError(t, err)
	bare, err := SortKey(token, ds)
	require.NoError(t, err)

	require.Equal(t, bare, appended)
}

func TestSortKeyNotSortableRevision(t *testing.T) {
	ds := unsortableHolder{datastoreUniqueID: testDatastoreID}
	token := mustToken(t, revisions.NewForTransactionID(42))

	t.Run("SortKey", func(t *testing.T) {
		_, err := SortKey(token, ds)
		require.Error(t, err)
		require.ErrorIs(t, err, ErrNotSortable)
	})

	t.Run("AppendSortKey returns dst unchanged", func(t *testing.T) {
		dst := []byte("prefix")
		out, err := AppendSortKey(dst, token, ds)
		require.ErrorIs(t, err, ErrNotSortable)
		require.Equal(t, []byte("prefix"), out)
	})

	t.Run("Compare", func(t *testing.T) {
		_, err := Compare(token, token, ds)
		require.ErrorIs(t, err, ErrNotSortable)
	})

	t.Run("Sort", func(t *testing.T) {
		tokens := []*v1.ZedToken{token, token}
		require.ErrorIs(t, Sort(tokens, ds), ErrNotSortable)
	})

	t.Run("single-element Sort still validates", func(t *testing.T) {
		require.ErrorIs(t, Sort([]*v1.ZedToken{token}, ds), ErrNotSortable)
	})
}

func TestSortKeyErrorCases(t *testing.T) {
	sortableDS := decoderFor(revisions.TransactionID)

	tcs := []struct {
		name       string
		token      *v1.ZedToken
		ds         RevisionHolder
		expectedIs error
	}{
		{
			name:       "nil token",
			token:      nil,
			ds:         sortableDS,
			expectedIs: ErrNilZedToken,
		},
		{
			name:  "malformed base64",
			token: &v1.ZedToken{Token: "!!!not-base64!!!"},
			ds:    sortableDS,
		},
		{
			name:  "base64 that is not a zedtoken",
			token: &v1.ZedToken{Token: "abc"},
			ds:    sortableDS,
		},
		{
			name: "mismatched datastore ID",
			token: func() *v1.ZedToken {
				encoded, err := newFromRevision(revisions.NewForTransactionID(42), "someotherdatastoreid", datalayer.NoSchemaHashInLegacyZedToken)
				require.NoError(t, err)
				return encoded
			}(),
			ds:         sortableDS,
			expectedIs: ErrDatastoreIDMismatch,
		},
	}

	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			_, err := SortKey(tc.token, tc.ds)
			require.Error(t, err)
			if tc.expectedIs != nil {
				require.ErrorIs(t, err, tc.expectedIs)
			}
			require.NotErrorIs(t, err, ErrNotSortable,
				"these failures are not a missing total order")
		})
	}
}

// TestSortKeyAcceptsLegacyEmptyDatastoreID pins the documented decision: a token with no datastore
// unique ID is keyed rather than refused.
func TestSortKeyAcceptsLegacyEmptyDatastoreID(t *testing.T) {
	ds := decoderFor(revisions.TransactionID)

	legacy, err := NewFromRevisionNoDatastoreID(revisions.NewForTransactionID(42), datalayer.NoSchemaHashInLegacyZedToken)
	require.NoError(t, err)

	decoded, err := DecodeRevision(legacy, ds)
	require.NoError(t, err)
	require.Equal(t, StatusLegacyEmptyDatastoreID, decoded.Status)

	legacyKey, err := SortKey(legacy, ds)
	require.NoError(t, err)

	// The key is the revision's, so it equals the key of a fully attributed token for the same
	// revision: the datastore ID prefix is not mixed in.
	currentKey, err := SortKey(mustToken(t, revisions.NewForTransactionID(42)), ds)
	require.NoError(t, err)
	require.Equal(t, currentKey, legacyKey)
}

// TestSortKeyAcceptsLegacyZookie covers the deprecated V1 zookie arm of the decoder, which also
// reports StatusLegacyEmptyDatastoreID.
func TestSortKeyAcceptsLegacyZookie(t *testing.T) {
	ds := decoderFor(revisions.TransactionID)

	// "CAESAggB" is the V1 zookie for revision 1; see decodeTests in zedtoken_test.go.
	key, err := SortKey(&v1.ZedToken{Token: "CAESAggB"}, ds)
	require.NoError(t, err)

	want, err := SortKey(mustToken(t, revisions.NewForTransactionID(1)), ds)
	require.NoError(t, err)
	require.Equal(t, want, key)
}

func TestSortKeyIgnoresSchemaHash(t *testing.T) {
	ds := decoderFor(revisions.TransactionID)
	rev := revisions.NewForTransactionID(42)

	withHashA, err := newFromRevision(rev, testDatastoreID, datalayer.SchemaHash("aaaaaaaaaaaaaaaa"))
	require.NoError(t, err)
	withHashB, err := newFromRevision(rev, testDatastoreID, datalayer.SchemaHash("bbbbbbbbbbbbbbbb"))
	require.NoError(t, err)

	// The encoded tokens differ...
	require.NotEqual(t, withHashA.Token, withHashB.Token)

	// ...but the sort keys do not, because the key is revision order alone.
	keyA, err := SortKey(withHashA, ds)
	require.NoError(t, err)
	keyB, err := SortKey(withHashB, ds)
	require.NoError(t, err)
	require.Equal(t, keyA, keyB)

	cmp, err := Compare(withHashA, withHashB, ds)
	require.NoError(t, err)
	require.Zero(t, cmp)
}

func TestCompare(t *testing.T) {
	ds := decoderFor(revisions.TransactionID)

	low := mustToken(t, revisions.NewForTransactionID(9))
	high := mustToken(t, revisions.NewForTransactionID(10))
	alsoLow := mustToken(t, revisions.NewForTransactionID(9))

	tcs := []struct {
		name     string
		a, b     *v1.ZedToken
		expected int
	}{
		{name: "less", a: low, b: high, expected: -1},
		{name: "greater", a: high, b: low, expected: 1},
		{name: "equal", a: low, b: alsoLow, expected: 0},
		{name: "identical", a: low, b: low, expected: 0},
	}

	for _, tc := range tcs {
		t.Run(tc.name, func(t *testing.T) {
			cmp, err := Compare(tc.a, tc.b, ds)
			require.NoError(t, err)
			require.Equal(t, tc.expected, cmp)
		})
	}
}

func TestCompareErrorsOnEitherSide(t *testing.T) {
	ds := decoderFor(revisions.TransactionID)
	good := mustToken(t, revisions.NewForTransactionID(1))

	_, err := Compare(nil, good, ds)
	require.ErrorIs(t, err, ErrNilZedToken)

	_, err = Compare(good, nil, ds)
	require.ErrorIs(t, err, ErrNilZedToken)
}

func TestSort(t *testing.T) {
	ds := decoderFor(revisions.TransactionID)

	t.Run("empty and single", func(t *testing.T) {
		require.NoError(t, Sort(nil, ds))
		require.NoError(t, Sort([]*v1.ZedToken{}, ds))
		require.NoError(t, Sort([]*v1.ZedToken{mustToken(t, revisions.NewForTransactionID(1))}, ds))
	})

	t.Run("orders oldest first", func(t *testing.T) {
		want := []uint64{0, 1, 9, 10, 100, 256, 1000, 1621538189028928000}
		tokens := make([]*v1.ZedToken, 0, len(want))
		for i := len(want) - 1; i >= 0; i-- {
			tokens = append(tokens, mustToken(t, revisions.NewForTransactionID(want[i])))
		}

		require.NoError(t, Sort(tokens, ds))

		for i, token := range tokens {
			decoded, err := DecodeRevision(token, ds)
			require.NoError(t, err)
			require.True(t, revisions.NewForTransactionID(want[i]).Equal(decoded.Revision),
				"position %d: want %d, got %s", i, want[i], decoded.Revision)
		}
	})

	t.Run("is stable for equal revisions", func(t *testing.T) {
		rev := revisions.NewForTransactionID(42)
		first, err := newFromRevision(rev, testDatastoreID, datalayer.SchemaHash("aaaaaaaaaaaaaaaa"))
		require.NoError(t, err)
		second, err := newFromRevision(rev, testDatastoreID, datalayer.SchemaHash("bbbbbbbbbbbbbbbb"))
		require.NoError(t, err)

		tokens := []*v1.ZedToken{first, second}
		require.NoError(t, Sort(tokens, ds))
		require.Equal(t, first.Token, tokens[0].Token)
		require.Equal(t, second.Token, tokens[1].Token)
	})

	t.Run("leaves the slice untouched on error", func(t *testing.T) {
		high := mustToken(t, revisions.NewForTransactionID(100))
		low := mustToken(t, revisions.NewForTransactionID(1))
		tokens := []*v1.ZedToken{high, low, nil}

		err := Sort(tokens, ds)
		require.ErrorIs(t, err, ErrNilZedToken)
		require.Equal(t, high, tokens[0])
		require.Equal(t, low, tokens[1])
		require.Nil(t, tokens[2])
	})
}

// TestSortKeyErrorsAreUnwrappable guards the wrapping, which is easy to break by switching a %w
// for a %v.
func TestSortKeyErrorsAreUnwrappable(t *testing.T) {
	_, err := SortKey(mustToken(t, revisions.NewForTransactionID(1)), unsortableHolder{datastoreUniqueID: testDatastoreID})
	require.Error(t, err)
	require.ErrorIs(t, err, ErrNotSortable)
	require.ErrorContains(t, err, "sort key")
	require.ErrorContains(t, err, "unsortableRevision", "the concrete type should be named")
}
