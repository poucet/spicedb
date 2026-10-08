package zedtoken

import (
	"bytes"
	"errors"
	"fmt"
	"slices"

	v1 "github.com/authzed/authzed-go/proto/authzed/api/v1"

	"github.com/authzed/spicedb/pkg/datastore"
)

// errSortKeyError wraps failures that happen while turning a zedtoken into a sort key. It is
// distinct from errDecodeError because a token can decode perfectly well and still have no sort
// key: see ErrNotSortable.
const errSortKeyError = "error computing zedtoken sort key: %w"

var (
	// ErrNotSortable is returned when the revision carried by a zedtoken is of a type that is only
	// partially ordered, and so has no sort key. It is a property of the datastore's revision type,
	// not of the individual token: every token from such a datastore reports it, and no token from
	// a totally ordered datastore ever does.
	//
	// Postgres is the case that exists today. Its revisions are transaction snapshots, and two
	// snapshots can be mutually uncomparable - neither before nor after the other - so there is no
	// order for bytes to preserve. See datastore.SortKeyRevision.
	//
	// Callers that must cope with any datastore should treat this as "sorting is unavailable here"
	// rather than as a failure of the token.
	ErrNotSortable = errors.New("zedtoken revision type is only partially ordered and has no sort key")

	// ErrDatastoreIDMismatch is returned when a zedtoken carries a datastore unique ID that does
	// not match the datastore it is being sorted against. Sort keys order revisions only within a
	// single datastore, so a key derived from a foreign token would compare against local keys
	// without meaning anything. Refusing is the only safe answer.
	ErrDatastoreIDMismatch = errors.New("zedtoken was issued by a different datastore")
)

// AppendSortKey appends the sort key for the revision inside encoded to dst and returns the
// extended buffer, like the builtin append. Pass a nil dst to get a key on its own.
//
// The key is derived purely from the revision. It deliberately does not mix in the schema hash or
// the datastore unique ID prefix: either would partition the key space ahead of the revision bytes
// and destroy revision order. Two zedtokens naming the same revision therefore produce byte-equal
// keys even when they were issued with different schema hashes.
//
// For any two zedtokens a and b issued by the same datastore, the keys guarantee:
//
//   - They sort in revision order. bytes.Compare of a's key and b's key is negative when a's
//     revision is less than b's, zero when they are equal, positive when it is greater.
//   - Neither key is a prefix of the other, so appending more bytes to each cannot reorder them. A
//     zedtoken sort key can therefore be one field of a longer key.
//
// The encoded token string gives neither. It is base64 of a protobuf whose byte order has nothing
// to do with revision order.
//
// Keys are comparable only between zedtokens from the same datastore, and are stable across
// SpiceDB versions, so they may be stored durably. Both properties are inherited directly from
// datastore.SortKeyRevision.AppendSortKey, which produces the bytes.
//
// Errors:
//
//   - ErrNotSortable, via errors.Is, when the datastore's revisions are only partially ordered.
//     Postgres is the case that exists today.
//   - ErrDatastoreIDMismatch, via errors.Is, when the token names a different datastore.
//   - A decode error for a nil, malformed or unrecognized token.
//
// Legacy tokens carrying no datastore unique ID are accepted. Such a token cannot be proven to
// have come from this datastore, because there is nothing to compare, but refusing it would make
// this API useless on every token issued before the datastore ID field existed and on every token
// from NewFromRevisionNoDatastoreID. The rest of the package already extends the same trust: a
// legacy token's revision is returned by DecodeRevision and used to read at, which is a far more
// consequential use than ordering it. Callers that need provenance should check
// DecodedRevision.Status themselves and reject StatusLegacyEmptyDatastoreID before sorting.
//
// AppendSortKey allocates only to grow dst. It keeps no state, so calls may be made concurrently,
// but dst belongs to the caller: concurrent calls must not share a dst backing array, and - as
// with the builtin append - a later call that reuses an earlier call's buffer may overwrite the
// earlier key.
func AppendSortKey(dst []byte, encoded *v1.ZedToken, ds RevisionHolder) ([]byte, error) {
	skr, err := sortKeyRevision(encoded, ds)
	if err != nil {
		return dst, err
	}

	return skr.AppendSortKey(dst), nil
}

// SortKey returns the sort key for the revision inside encoded, on its own. It is shorthand for
// AppendSortKey(nil, encoded, ds); see that function for the guarantees the key carries, the
// treatment of legacy tokens, and the errors returned.
func SortKey(encoded *v1.ZedToken, ds RevisionHolder) ([]byte, error) {
	return AppendSortKey(nil, encoded, ds)
}

// Compare orders two zedtokens by the revisions they carry, returning -1 if a is before b, 0 if
// they name the same revision, and +1 if a is after b.
//
// It compares the sort keys, so its answer is by construction the same one a caller gets by
// sorting keys from SortKey, and the same one the decoded revisions give through LessThan, Equal
// and GreaterThan.
//
// Zero does not mean the tokens are identical: two tokens naming one revision with different
// schema hashes compare equal, because the sort key is revision order alone.
//
// Both tokens are resolved against ds, and the errors are those of AppendSortKey. A caller sorting
// more than a handful of tokens should use Sort, or compute each key once with SortKey and compare
// with bytes.Compare, rather than paying a decode per comparison.
func Compare(a, b *v1.ZedToken, ds RevisionHolder) (int, error) {
	aKey, err := SortKey(a, ds)
	if err != nil {
		return 0, err
	}

	bKey, err := SortKey(b, ds)
	if err != nil {
		return 0, err
	}

	return bytes.Compare(aKey, bKey), nil
}

// Sort sorts tokens in place into revision order, oldest first.
//
// Each token is decoded and keyed once, rather than once per comparison, and the sort is stable,
// so tokens naming the same revision keep their relative order. Every token is keyed before
// anything is moved, so if any one of them cannot be keyed, tokens is left untouched and the
// error is returned; the errors are those of AppendSortKey. A one-element slice is validated too,
// rather than trivially accepted, so that it cannot quietly admit a token a longer slice rejects.
func Sort(tokens []*v1.ZedToken, ds RevisionHolder) error {
	type keyed struct {
		token *v1.ZedToken
		key   []byte
	}

	decorated := make([]keyed, 0, len(tokens))
	for _, token := range tokens {
		key, err := SortKey(token, ds)
		if err != nil {
			return err
		}
		decorated = append(decorated, keyed{token: token, key: key})
	}

	if len(decorated) < 2 {
		return nil
	}

	slices.SortStableFunc(decorated, func(a, b keyed) int {
		return bytes.Compare(a.key, b.key)
	})

	sorted := make([]*v1.ZedToken, 0, len(decorated))
	for _, d := range decorated {
		sorted = append(sorted, d.token)
	}
	copy(tokens, sorted)

	return nil
}

// sortKeyRevision decodes encoded and returns its revision as a datastore.SortKeyRevision, or an
// error explaining why no sort key exists for it.
//
// It goes through DecodeRevision rather than a narrower path of its own so that the revision
// parsing and the datastore ID check cannot drift from the ones every other caller sees. The only
// thing it discards is the schema hash, which has no bearing on order.
func sortKeyRevision(encoded *v1.ZedToken, ds RevisionHolder) (datastore.SortKeyRevision, error) {
	decoded, err := DecodeRevision(encoded, ds)
	if err != nil {
		return nil, err
	}

	switch decoded.Status {
	case StatusValid:
		// Provenance proven: the token names this datastore.

	case StatusLegacyEmptyDatastoreID:
		// Provenance unprovable but accepted; see AppendSortKey's doc comment.

	case StatusMismatchedDatastoreID:
		return nil, fmt.Errorf(errSortKeyError, ErrDatastoreIDMismatch)

	default:
		return nil, fmt.Errorf(errSortKeyError, fmt.Errorf("unexpected zedtoken status: %d", decoded.Status))
	}

	skr, ok := decoded.Revision.(datastore.SortKeyRevision)
	if !ok {
		return nil, fmt.Errorf(errSortKeyError, fmt.Errorf("%w: revision of type %T", ErrNotSortable, decoded.Revision))
	}

	return skr, nil
}
