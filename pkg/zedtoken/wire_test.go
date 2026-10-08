package zedtoken

import (
	"bytes"
	"context"
	"slices"
	"strconv"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	v1 "github.com/authzed/authzed-go/proto/authzed/api/v1"

	"github.com/authzed/spicedb/internal/datastore/revisions"
	"github.com/authzed/spicedb/pkg/datalayer"
	"github.com/authzed/spicedb/pkg/datastore"
)

// This file covers the wiring that stamps ZedToken.sort_key at token-issue time. It is about the
// plumbing - does a minted token carry a key, is it the right key, does it survive the wire - and
// not about the correctness of any particular revision type's encoding, which each datastore's own
// tests own.
//
// Every identifier here is prefixed `wire` so this file can sit alongside other test files in
// package zedtoken without colliding with their fixtures.

const wireDatastoreID = "wiretestdatastore"

// wireUnsortableRevision is a datastore.Revision that does NOT implement
// datastore.SortKeyRevision, standing in for any revision type that cannot be ordered: one that is
// only partially ordered, or one that simply predates the interface. The point is the absence of
// the method, so a local type demonstrates it without dragging a datastore implementation in here.
type wireUnsortableRevision uint64

func (wr wireUnsortableRevision) ByteSortable() bool { return false }

func (wr wireUnsortableRevision) Equal(rhs datastore.Revision) bool {
	other, ok := rhs.(wireUnsortableRevision)
	return ok && wr == other
}

func (wr wireUnsortableRevision) GreaterThan(rhs datastore.Revision) bool {
	other, ok := rhs.(wireUnsortableRevision)
	return ok && wr > other
}

func (wr wireUnsortableRevision) LessThan(rhs datastore.Revision) bool {
	other, ok := rhs.(wireUnsortableRevision)
	return ok && wr < other
}

func (wr wireUnsortableRevision) String() string {
	return strconv.FormatUint(uint64(wr), 10)
}

var _ datastore.Revision = wireUnsortableRevision(0)

// wireUnsortableHolder is a RevisionHolder whose revisions have no sort key.
type wireUnsortableHolder struct{}

func (wireUnsortableHolder) UniqueID(_ context.Context) (string, error) {
	return wireDatastoreID, nil
}

func (wireUnsortableHolder) RevisionFromString(s string) (datastore.Revision, error) {
	parsed, err := strconv.ParseUint(s, 10, 64)
	if err != nil {
		return nil, err
	}
	return wireUnsortableRevision(parsed), nil
}

var _ RevisionHolder = wireUnsortableHolder{}

func wireHolderFor(kind revisions.RevisionKind) revisions.CommonDecoder {
	return revisions.CommonDecoder{Kind: kind, DatastoreUniqueID: wireDatastoreID}
}

func wireMustHLC(t *testing.T, s string) datastore.Revision {
	t.Helper()
	rev, err := revisions.HLCRevisionFromString(s)
	require.NoError(t, err)
	return rev
}

// wireRevisionCase is one datastore's revision type, with revisions already in ascending order.
type wireRevisionCase struct {
	name      string
	kind      revisions.RevisionKind
	keyLength int
	ascending []datastore.Revision
}

func wireRevisionCases(t *testing.T) []wireRevisionCase {
	t.Helper()

	return []wireRevisionCase{
		{
			// mysql
			name:      "transaction ID",
			kind:      revisions.TransactionID,
			keyLength: 8,
			ascending: []datastore.Revision{
				revisions.NewForTransactionID(0),
				revisions.NewForTransactionID(1),
				revisions.NewForTransactionID(2),
				revisions.NewForTransactionID(9),
				revisions.NewForTransactionID(10),
				revisions.NewForTransactionID(100),
				revisions.NewForTransactionID(1000),
				revisions.NewForTransactionID(1621538189028928000),
			},
		},
		{
			// spanner, memdb
			name:      "timestamp",
			kind:      revisions.Timestamp,
			keyLength: 8,
			ascending: []datastore.Revision{
				revisions.NewForTimestamp(0),
				revisions.NewForTimestamp(1),
				revisions.NewForTimestamp(9),
				revisions.NewForTimestamp(10),
				revisions.NewForTime(time.Unix(1, 0)),
				revisions.NewForTime(time.Unix(1_000_000, 0)),
				revisions.NewForTimestamp(1621538189028928000),
			},
		},
		{
			// crdb
			name:      "hybrid logical clock",
			kind:      revisions.HybridLogicalClock,
			keyLength: 12,
			ascending: []datastore.Revision{
				wireMustHLC(t, "1234"),
				wireMustHLC(t, "1234.0000000001"),
				wireMustHLC(t, "1235"),
				wireMustHLC(t, "9999"),
				wireMustHLC(t, "10000"),
				wireMustHLC(t, "1621538189028928000"),
			},
		},
	}
}

// TestNewFromRevisionSetsSortKey is the headline: a token minted through the real public path
// carries a sort key, of the width its revision type promises, and those keys sort in revision
// order. This is what a client in any language gets to sort on.
func TestNewFromRevisionSetsSortKey(t *testing.T) {
	t.Parallel()

	for _, tc := range wireRevisionCases(t) {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			holder := wireHolderFor(tc.kind)
			ctx := t.Context()

			tokens := make([]*v1.ZedToken, 0, len(tc.ascending))
			for _, rev := range tc.ascending {
				token, err := NewFromRevision(ctx, rev, datalayer.NoSchemaHashInLegacyZedToken, holder)
				require.NoError(t, err)
				require.NotEmpty(t, token.GetSortKey(), "revision %s minted a token with no sort key", rev)
				require.Len(t, token.GetSortKey(), tc.keyLength, "revision %s", rev)
				tokens = append(tokens, token)
			}

			// The fixtures are already ascending, so the keys must already be ascending.
			for i := 1; i < len(tokens); i++ {
				require.Negative(t,
					bytes.Compare(tokens[i-1].GetSortKey(), tokens[i].GetSortKey()),
					"sort key for %s did not sort below %s", tc.ascending[i-1], tc.ascending[i])
			}

			// And shuffling then sorting by key must put them back. Derive the expectation from
			// LessThan rather than from the fixture order, so a key and a hand-written list cannot
			// be wrong in the same way.
			shuffled := slices.Clone(tokens)
			slices.Reverse(shuffled)
			slices.SortFunc(shuffled, func(a, b *v1.ZedToken) int {
				return bytes.Compare(a.GetSortKey(), b.GetSortKey())
			})
			require.Equal(t, tokens, shuffled, "sorting by sort key did not recover revision order")
		})
	}
}

// TestNewFromRevisionSortKeyIsExactlyTheRevisionKey pins that the token carries the revision's own
// encoding and nothing else: no schema hash, no datastore ID prefix, no framing. Either of those
// would partition the key space ahead of the revision bytes and destroy revision order.
func TestNewFromRevisionSortKeyIsExactlyTheRevisionKey(t *testing.T) {
	t.Parallel()

	for _, tc := range wireRevisionCases(t) {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			holder := wireHolderFor(tc.kind)
			for _, rev := range tc.ascending {
				token, err := NewFromRevision(t.Context(), rev, datalayer.NoSchemaHashInLegacyZedToken, holder)
				require.NoError(t, err)

				skr, ok := rev.(datastore.SortKeyRevision)
				require.True(t, ok, "fixture %s is not a SortKeyRevision", rev)
				require.Equal(t, skr.AppendSortKey(nil), token.GetSortKey(), "revision %s", rev)
			}
		})
	}
}

// TestNewFromRevisionSortKeyIgnoresSchemaHash is the other half of the same rule: two tokens for
// one revision issued with different schema hashes must carry byte-identical keys, or the schema
// hash has leaked into the order.
func TestNewFromRevisionSortKeyIgnoresSchemaHash(t *testing.T) {
	t.Parallel()

	holder := wireHolderFor(revisions.TransactionID)
	rev := revisions.NewForTransactionID(42)
	ctx := t.Context()

	withoutHash, err := NewFromRevision(ctx, rev, datalayer.NoSchemaHashInLegacyZedToken, holder)
	require.NoError(t, err)

	withHash, err := NewFromRevision(ctx, rev, datalayer.SchemaHash("some-schema-hash"), holder)
	require.NoError(t, err)

	require.Equal(t, withoutHash.GetSortKey(), withHash.GetSortKey())
	require.NotEqual(t, withoutHash.GetToken(), withHash.GetToken(),
		"fixture is not exercising two genuinely different tokens")
}

// TestNewFromRevisionUnsortableIsNotAnError is the regression this wiring most needs. A revision
// type that cannot produce a key must yield a keyless token and NO error: failing here would turn
// an optional interface into a hard dependency and would fail Writes on such a datastore.
func TestNewFromRevisionUnsortableIsNotAnError(t *testing.T) {
	t.Parallel()

	token, err := NewFromRevision(t.Context(), wireUnsortableRevision(7),
		datalayer.NoSchemaHashInLegacyZedToken, wireUnsortableHolder{})

	require.NoError(t, err, "an unsortable revision must not fail token issuance")
	require.NotNil(t, token)
	require.Empty(t, token.GetSortKey())
	require.NotEmpty(t, token.GetToken(), "the token itself must still be usable")

	// And it must still decode back to the revision it names, which is the whole point of saying
	// an unsortable token is still a good token.
	decoded, err := DecodeRevision(token, wireUnsortableHolder{})
	require.NoError(t, err)
	require.True(t, decoded.Revision.Equal(wireUnsortableRevision(7)))
}

// TestNoRevisionYieldsNoSortKey covers datastore.NoRevision, whose nilRevision sentinel implements
// no sort key. It reaches the same path and must be keyless rather than panicking.
func TestNoRevisionYieldsNoSortKey(t *testing.T) {
	t.Parallel()

	token, err := NewFromRevisionNoDatastoreID(datastore.NoRevision, datalayer.NoSchemaHashInLegacyZedToken)
	require.NoError(t, err)
	require.Empty(t, token.GetSortKey())
}

// TestAllMintingEntryPointsSetSortKey guards the funnel. Every exported way to mint a token goes
// through newFromRevision, so every one of them must come out keyed; if someone adds a path that
// bypasses the funnel, this is the test that should start looking incomplete.
func TestAllMintingEntryPointsSetSortKey(t *testing.T) {
	t.Parallel()

	rev := revisions.NewForTransactionID(123)
	want := rev.AppendSortKey(nil)

	t.Run("NewFromRevision", func(t *testing.T) {
		t.Parallel()
		token, err := NewFromRevision(t.Context(), rev, datalayer.NoSchemaHashInLegacyZedToken,
			wireHolderFor(revisions.TransactionID))
		require.NoError(t, err)
		require.Equal(t, want, token.GetSortKey())
	})

	t.Run("NewFromRevisionNoDatastoreID", func(t *testing.T) {
		t.Parallel()
		token, err := NewFromRevisionNoDatastoreID(rev, datalayer.NoSchemaHashInLegacyZedToken)
		require.NoError(t, err)
		require.Equal(t, want, token.GetSortKey(),
			"a legacy token with no datastore ID must still be sortable")
	})

	t.Run("MustNewFromRevisionForTesting", func(t *testing.T) {
		t.Parallel()
		token := MustNewFromRevisionForTesting(rev, datalayer.NoSchemaHashInLegacyZedToken)
		require.Equal(t, want, token.GetSortKey())
	})
}

// TestEncodeDoesNotSetSortKey documents the one exported path that deliberately cannot key a
// token. Encode takes a DecodedZedToken, which holds the revision only as a string, and a string
// is not enough to produce an order-preserving encoding. This is a documented limitation, not an
// oversight, so it is pinned rather than left to drift.
func TestEncodeDoesNotSetSortKey(t *testing.T) {
	t.Parallel()

	rev := revisions.NewForTransactionID(5)

	keyed, err := NewFromRevisionNoDatastoreID(rev, datalayer.NoSchemaHashInLegacyZedToken)
	require.NoError(t, err)
	require.NotEmpty(t, keyed.GetSortKey())

	decoded, err := Decode(keyed)
	require.NoError(t, err)

	reEncoded, err := Encode(decoded)
	require.NoError(t, err)

	require.Equal(t, keyed.GetToken(), reEncoded.GetToken(), "the opaque token must round trip")
	require.Empty(t, reEncoded.GetSortKey(), "Encode cannot know the revision, so it cannot key")
}

// TestSortKeySurvivesTheWire checks the field actually crosses a process boundary, under both the
// standard codec and vtproto, which SpiceDB uses on hot paths. A field that is set in memory and
// dropped on the wire would be the worst kind of pass.
func TestSortKeySurvivesTheWire(t *testing.T) {
	t.Parallel()

	token, err := NewFromRevision(t.Context(), revisions.NewForTransactionID(77),
		datalayer.NoSchemaHashInLegacyZedToken, wireHolderFor(revisions.TransactionID))
	require.NoError(t, err)
	require.NotEmpty(t, token.GetSortKey())

	t.Run("proto", func(t *testing.T) {
		t.Parallel()
		raw, err := proto.Marshal(token)
		require.NoError(t, err)

		var got v1.ZedToken
		require.NoError(t, proto.Unmarshal(raw, &got))
		require.Equal(t, token.GetSortKey(), got.GetSortKey())
		require.Equal(t, token.GetToken(), got.GetToken())
	})

	t.Run("vtproto", func(t *testing.T) {
		t.Parallel()
		raw, err := token.MarshalVT()
		require.NoError(t, err)

		var got v1.ZedToken
		require.NoError(t, got.UnmarshalVT(raw))
		require.Equal(t, token.GetSortKey(), got.GetSortKey())
		require.Equal(t, token.GetToken(), got.GetToken())
	})

	t.Run("codecs agree", func(t *testing.T) {
		t.Parallel()
		standard, err := proto.Marshal(token)
		require.NoError(t, err)
		vt, err := token.MarshalVT()
		require.NoError(t, err)
		require.Equal(t, standard, vt, "vtproto and proto must produce identical bytes")
		require.Equal(t, len(standard), token.SizeVT(), "SizeVT must account for the sort key")
	})
}

// TestMissingSortKeySortsFirst records the hazard that drives the client documentation. A token
// with no key does not raise anything under a naive sort: it sorts FIRST, claiming to be the
// oldest, and returns a plausible wrong answer. Clients must partition on emptiness before
// sorting. If this test ever fails, the client guidance needs rewriting, not the test.
func TestMissingSortKeySortsFirst(t *testing.T) {
	t.Parallel()

	ctx := t.Context()
	holder := wireHolderFor(revisions.TransactionID)

	oldest, err := NewFromRevision(ctx, revisions.NewForTransactionID(1), datalayer.NoSchemaHashInLegacyZedToken, holder)
	require.NoError(t, err)
	newest, err := NewFromRevision(ctx, revisions.NewForTransactionID(99), datalayer.NoSchemaHashInLegacyZedToken, holder)
	require.NoError(t, err)

	// A token that lost its key, exactly as a client gets back after persisting only the opaque
	// string and rebuilding the message from it.
	restored := &v1.ZedToken{Token: newest.GetToken()}
	require.Empty(t, restored.GetSortKey())

	all := []*v1.ZedToken{oldest, restored, newest}
	slices.SortStableFunc(all, func(a, b *v1.ZedToken) int {
		return bytes.Compare(a.GetSortKey(), b.GetSortKey())
	})

	require.Same(t, restored, all[0],
		"a keyless token is expected to sort first; the client docs warn about exactly this")
	require.Empty(t, all[0].GetSortKey())

	// The safe recipe: partition first, then sort only what can be ordered.
	sortable := make([]*v1.ZedToken, 0, len(all))
	unsortable := make([]*v1.ZedToken, 0, len(all))
	for _, token := range all {
		if len(token.GetSortKey()) == 0 {
			unsortable = append(unsortable, token)
			continue
		}
		sortable = append(sortable, token)
	}
	require.Len(t, unsortable, 1)
	require.Len(t, sortable, 2)
	require.Negative(t, bytes.Compare(sortable[0].GetSortKey(), sortable[1].GetSortKey()))
}

// TestRoundTripThroughTokenStringLosesTheKey is the in-code form of the migration warning: storing
// only the opaque string, which is what the SpiceDB docs tell callers to put in a text column,
// silently drops the sort key.
func TestRoundTripThroughTokenStringLosesTheKey(t *testing.T) {
	t.Parallel()

	token, err := NewFromRevision(t.Context(), revisions.NewForTransactionID(11),
		datalayer.NoSchemaHashInLegacyZedToken, wireHolderFor(revisions.TransactionID))
	require.NoError(t, err)
	require.NotEmpty(t, token.GetSortKey())

	persisted := token.GetToken() // what a caller puts in a `text` column
	restored := &v1.ZedToken{Token: persisted}

	require.Equal(t, token.GetToken(), restored.GetToken())
	require.Empty(t, restored.GetSortKey(), "persisting only the string loses the key")

	// Storing both columns round trips exactly, which is the guidance that has to land.
	persistedKey := token.GetSortKey()
	restoredBoth := &v1.ZedToken{Token: persisted, SortKey: persistedKey}
	require.Equal(t, token.GetSortKey(), restoredBoth.GetSortKey())
}
