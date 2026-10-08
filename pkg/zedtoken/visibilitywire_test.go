package zedtoken

import (
	"context"
	"encoding/base64"
	"testing"
	"time"

	"github.com/stretchr/testify/require"
	"google.golang.org/protobuf/proto"

	v1 "github.com/authzed/authzed-go/proto/authzed/api/v1"

	"github.com/authzed/spicedb/internal/datastore/postgres"
	"github.com/authzed/spicedb/internal/datastore/revisions"
	"github.com/authzed/spicedb/pkg/datalayer"
	"github.com/authzed/spicedb/pkg/datastore"
	implv1 "github.com/authzed/spicedb/pkg/proto/impl/v1"
)

// pgRevision builds a real postgresRevision from a snapshot spec, by way of the revision string
// format postgresRevision.String() produces. The relative encoding mirrors MarshalBinary.
func pgRevision(t *testing.T, xmin, xmax uint64, xips ...uint64) datastore.Revision {
	t.Helper()

	relativeXips := make([]int64, 0, len(xips))
	for _, xip := range xips {
		relativeXips = append(relativeXips, int64(xip)-int64(xmin)) //nolint:gosec // test fixture values are small.
	}

	encoded, err := proto.Marshal(&implv1.PostgresRevision{
		Xmin:         xmin,
		RelativeXmax: int64(xmax) - int64(xmin), //nolint:gosec // test fixture values are small.
		RelativeXips: relativeXips,
	})
	require.NoError(t, err)

	rev, err := postgres.ParseRevisionString(base64.StdEncoding.EncodeToString(encoded))
	require.NoError(t, err)
	return rev
}

// pgHolder is a RevisionHolder over real Postgres revisions.
type pgHolder struct{ id string }

func (h pgHolder) UniqueID(_ context.Context) (string, error) { return h.id, nil }
func (h pgHolder) RevisionFromString(s string) (datastore.Revision, error) {
	return postgres.ParseRevisionString(s)
}

const wireTestDatastoreID = "c0ffee00-1111-2222-3333-444444444444"

func pgToken(t *testing.T, rev datastore.Revision) *v1.ZedToken {
	t.Helper()
	token, err := NewFromRevision(context.Background(), rev, datalayer.NoSchemaHashInLegacyZedToken, pgHolder{wireTestDatastoreID})
	require.NoError(t, err)
	return token
}

// TestPostgresTokensCarryVisibility pins that a real Postgres token leaves the server with the
// field populated. Without this the Python client has nothing to compare, and every other test in
// this package would still pass.
func TestPostgresTokensCarryVisibility(t *testing.T) {
	token := pgToken(t, pgRevision(t, 1, 5, 2, 3))

	require.NotNil(t, token.Visibility, "a Postgres token must carry a visible set")
	require.Equal(t, uint64(2), token.Visibility.Floor)
	require.Equal(t, uint64(5), token.Visibility.Ceiling)
	require.Equal(t, []uint64{0, 1}, token.Visibility.ExceptionOffsets,
		"exceptions 2 and 3 are offsets 0 and 1 from floor 2")
}

// TestTotallyOrderedTokensOmitVisibility pins the other half of the contract. Absence is the
// signal that sort_key alone is a sound basis for causality, so a totally ordered datastore must
// not set the field, and must not pay for it on every response either.
func TestTotallyOrderedTokensOmitVisibility(t *testing.T) {
	for _, tc := range []struct {
		name string
		kind revisions.RevisionKind
		rev  datastore.Revision
	}{
		{"transaction id", revisions.TransactionID, revisions.NewForTransactionID(42)},
		{"timestamp", revisions.Timestamp, revisions.NewForTime(time.Unix(1700000000, 0))},
	} {
		t.Run(tc.name, func(t *testing.T) {
			holder := revisions.CommonDecoder{Kind: tc.kind, DatastoreUniqueID: wireTestDatastoreID}
			token, err := NewFromRevision(context.Background(), tc.rev, datalayer.NoSchemaHashInLegacyZedToken, holder)
			require.NoError(t, err)

			require.Nil(t, token.Visibility,
				"a totally ordered revision must not ship a visible set; its absence is what tells a client sort_key is enough")
			require.NotEmpty(t, token.SortKey, "but it must still ship a sort key")
		})
	}
}

// TestVisibilityRoundTripsThroughTheWire proves the field survives real serialisation, including
// vtproto, which SpiceDB uses and which is generated separately from the standard codec.
func TestVisibilityRoundTripsThroughTheWire(t *testing.T) {
	token := pgToken(t, pgRevision(t, 10, 14, 11, 13))

	wire, err := proto.Marshal(token)
	require.NoError(t, err)

	var decoded v1.ZedToken
	require.NoError(t, proto.Unmarshal(wire, &decoded))
	require.Equal(t, token.Visibility.Floor, decoded.Visibility.Floor)
	require.Equal(t, token.Visibility.Ceiling, decoded.Visibility.Ceiling)
	require.Equal(t, token.Visibility.ExceptionOffsets, decoded.Visibility.ExceptionOffsets)

	vtWire, err := token.MarshalVT()
	require.NoError(t, err)
	require.Equal(t, wire, vtWire, "vtproto must agree with the standard codec byte for byte")

	var vtDecoded v1.ZedToken
	require.NoError(t, vtDecoded.UnmarshalVT(wire))
	require.Equal(t, token.Visibility.ExceptionOffsets, vtDecoded.Visibility.ExceptionOffsets)
}

// TestVisibilityIsIdenticalForEqualRevisions pins at the wire boundary the property a remote
// caller depends on: two Equal revisions with different internal representations must ship
// identical visible sets, or Python will call them concurrent while Go calls them equal.
func TestVisibilityIsIdenticalForEqualRevisions(t *testing.T) {
	// Equal under compare, and sharing no part of their representation.
	compact := pgToken(t, pgRevision(t, 1, 1))
	expanded := pgToken(t, pgRevision(t, 1, 5, 1, 2, 3, 4))

	require.True(t, compact.Visibility.Floor == expanded.Visibility.Floor &&
		compact.Visibility.Ceiling == expanded.Visibility.Ceiling,
		"Equal revisions must project identically, got %+v and %+v", compact.Visibility, expanded.Visibility)
	require.Equal(t, compact.Visibility.ExceptionOffsets, expanded.Visibility.ExceptionOffsets)
}

// TestVisibilitySizeScalesWithWritesInFlight measures the cost of the only unbounded field on a
// token, because that measurement is what justified evaluating causality on the client rather than
// behind an RPC.
//
// It also measures the token string, which already carries the same in-flight list in its own
// encoding, to show the field is a second copy of a cost the token was already paying rather than
// a new exposure.
func TestVisibilitySizeScalesWithWritesInFlight(t *testing.T) {
	for _, inFlight := range []int{0, 1, 8, 64, 500} {
		xips := make([]uint64, 0, inFlight)
		for i := range inFlight {
			// A settled transaction between each in-flight one, so canonicalize cannot collapse
			// the run and the worst case is really measured.
			xips = append(xips, uint64(1+2*i)) //nolint:gosec // loop bound is a small literal.
		}
		rev := pgRevision(t, 0, uint64(2*inFlight+2), xips...) //nolint:gosec // loop bound is a small literal.
		token := pgToken(t, rev)

		require.Len(t, token.Visibility.ExceptionOffsets, inFlight)

		visibilityBytes, err := proto.Marshal(token.Visibility)
		require.NoError(t, err)
		wholeToken, err := proto.Marshal(token)
		require.NoError(t, err)

		t.Logf("%3d writes in flight: visibility %4d bytes, token string %4d bytes, whole message %4d bytes",
			inFlight, len(visibilityBytes), len(token.Token), len(wholeToken))
	}
}

// TestVisibilityUnsupportedRevisionIsNotAnError pins that token issuance never fails because a
// revision type declines the optional interface. Failing a Write over a missing optional
// capability would be a far worse outcome than a token that cannot answer causal questions.
func TestVisibilityUnsupportedRevisionIsNotAnError(t *testing.T) {
	holder := revisions.CommonDecoder{Kind: revisions.TransactionID, DatastoreUniqueID: wireTestDatastoreID}

	token, err := NewFromRevision(context.Background(), datastore.NoRevision, datalayer.NoSchemaHashInLegacyZedToken, holder)
	require.NoError(t, err, "NoRevision must still mint a token")
	require.Nil(t, token.Visibility)
}

// TestEncodeDoesNotSetVisibility pins that Encode stays incapable of this, for the same reason it
// cannot set a sort key: it receives the revision as a string, and a string is not a visible set.
func TestEncodeDoesNotSetVisibility(t *testing.T) {
	encoded, err := Encode(&implv1.DecodedZedToken{
		VersionOneof: &implv1.DecodedZedToken_V1{
			V1: &implv1.DecodedZedToken_V1ZedToken{Revision: "1234"},
		},
	})
	require.NoError(t, err)
	require.Nil(t, encoded.Visibility)
	require.Nil(t, encoded.SortKey)
}
