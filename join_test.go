package dataframe

import (
	"errors"
	"hash/maphash"
	"reflect"
	"slices"
	"testing"

	"github.com/joeychilson/dataframe/series"
)

func TestUsingJoins(t *testing.T) {
	leftKeys := series.FromOptionals([]series.Optional[int]{series.Some(1), series.Some(2), series.Some(2), series.None[int]()})
	rightKeys := series.FromOptionals([]series.Optional[int]{series.Some(2), series.Some(2), series.Some(3), series.None[int]()})
	left, err := New(ColumnFromSeries("key", leftKeys), Column("left", []string{"a", "b", "c", "d"}))
	if err != nil {
		t.Fatal(err)
	}
	right, err := New(ColumnFromSeries("key", rightKeys), Column("right", []string{"x", "y", "z", "n"}))
	if err != nil {
		t.Fatal(err)
	}
	key := Using[int]("key")

	inner, err := left.InnerJoin(right, key)
	if err != nil {
		t.Fatal(err)
	}
	assertJoinColumns(t, inner,
		[]series.Optional[int]{series.Some(2), series.Some(2), series.Some(2), series.Some(2)},
		[]series.Optional[string]{series.Some("b"), series.Some("b"), series.Some("c"), series.Some("c")},
		[]series.Optional[string]{series.Some("x"), series.Some("y"), series.Some("x"), series.Some("y")},
	)

	leftJoined, err := left.LeftJoin(right, key)
	if err != nil {
		t.Fatal(err)
	}
	assertJoinColumns(t, leftJoined,
		[]series.Optional[int]{series.Some(1), series.Some(2), series.Some(2), series.Some(2), series.Some(2), series.None[int]()},
		[]series.Optional[string]{series.Some("a"), series.Some("b"), series.Some("b"), series.Some("c"), series.Some("c"), series.Some("d")},
		[]series.Optional[string]{series.None[string](), series.Some("x"), series.Some("y"), series.Some("x"), series.Some("y"), series.None[string]()},
	)
	if !leftJoined.Schema()[2].Nullable {
		t.Fatal("right output of LeftJoin is not nullable")
	}

	rightJoined, err := left.RightJoin(right, key)
	if err != nil {
		t.Fatal(err)
	}
	assertJoinColumns(t, rightJoined,
		[]series.Optional[int]{series.Some(2), series.Some(2), series.Some(2), series.Some(2), series.Some(3), series.None[int]()},
		[]series.Optional[string]{series.Some("b"), series.Some("c"), series.Some("b"), series.Some("c"), series.None[string](), series.None[string]()},
		[]series.Optional[string]{series.Some("x"), series.Some("x"), series.Some("y"), series.Some("y"), series.Some("z"), series.Some("n")},
	)
	if !rightJoined.Schema()[1].Nullable || rightJoined.Schema()[2].Nullable {
		t.Fatalf("RightJoin schema = %#v", rightJoined.Schema())
	}

	full, err := left.FullJoin(right, key)
	if err != nil {
		t.Fatal(err)
	}
	assertJoinColumns(t, full,
		[]series.Optional[int]{series.Some(1), series.Some(2), series.Some(2), series.Some(2), series.Some(2), series.None[int](), series.Some(3), series.None[int]()},
		[]series.Optional[string]{series.Some("a"), series.Some("b"), series.Some("b"), series.Some("c"), series.Some("c"), series.Some("d"), series.None[string](), series.None[string]()},
		[]series.Optional[string]{series.None[string](), series.Some("x"), series.Some("y"), series.Some("x"), series.Some("y"), series.None[string](), series.Some("z"), series.Some("n")},
	)
	if !full.Schema()[1].Nullable || !full.Schema()[2].Nullable {
		t.Fatalf("FullJoin schema = %#v", full.Schema())
	}
}

func TestOnJoinAndJoinOrder(t *testing.T) {
	left, err := New(Column("left_id", []int{0, 1}), Column("left_value", []string{"a", "b"}))
	if err != nil {
		t.Fatal(err)
	}
	right, err := New(Column("right_id", []int{0, 1, 2}), Column("right_value", []string{"x", "y", "z"}))
	if err != nil {
		t.Fatal(err)
	}
	joined, err := left.InnerJoin(right, On(series.New([]int{2, 2}), series.New([]int{2, 2, 3})))
	if err != nil {
		t.Fatal(err)
	}
	if got := joined.Names(); !slices.Equal(got, []string{"left_id", "left_value", "right_id", "right_value"}) {
		t.Fatalf("names = %v", got)
	}
	leftIDs, _ := joined.Column[int]("left_id")
	rightIDs, _ := joined.Column[int]("right_id")
	if !slices.Equal(leftIDs.Values(), []int{0, 0, 1, 1}) || !slices.Equal(rightIDs.Values(), []int{0, 1, 0, 1}) {
		t.Fatalf("row pairs = %v, %v", leftIDs.Values(), rightIDs.Values())
	}
}

func TestSemiAndAntiJoin(t *testing.T) {
	left, err := New(Column("key", []int{1, 2, 2, 4}), Column("value", []string{"a", "b", "c", "d"}))
	if err != nil {
		t.Fatal(err)
	}
	right, err := New(Column("key", []int{2, 3}))
	if err != nil {
		t.Fatal(err)
	}
	semi, err := left.SemiJoin(right, Using[int]("key"))
	if err != nil {
		t.Fatal(err)
	}
	anti, err := left.AntiJoin(right, Using[int]("key"))
	if err != nil {
		t.Fatal(err)
	}
	semiValues, _ := semi.Column[string]("value")
	antiValues, _ := anti.Column[string]("value")
	if !slices.Equal(semiValues.Values(), []string{"b", "c"}) || !slices.Equal(antiValues.Values(), []string{"a", "d"}) {
		t.Fatalf("semi = %v, anti = %v", semiValues.Values(), antiValues.Values())
	}
}

func TestUsingColumnsAndCustomHasher(t *testing.T) {
	left, err := New(Column("left_key", [][]int{{1}, {2}}), Column("left", []string{"a", "b"}))
	if err != nil {
		t.Fatal(err)
	}
	right, err := New(Column("right_key", [][]int{{2}, {3}}), Column("right", []string{"x", "y"}))
	if err != nil {
		t.Fatal(err)
	}
	joined, err := left.InnerJoin(right, UsingColumnsBy[[]int]("key", "left_key", "right_key", intSliceHasher{}))
	if err != nil {
		t.Fatal(err)
	}
	if got := joined.Names(); !slices.Equal(got, []string{"key", "left", "right"}) {
		t.Fatalf("names = %v", got)
	}
	keys, err := joined.Column[[]int]("key")
	if err != nil {
		t.Fatal(err)
	}
	if got := keys.Values(); !reflect.DeepEqual(got, [][]int{{2}}) {
		t.Fatalf("keys = %v", got)
	}

	sameNamedLeft, err := New(Column("key", [][]int{{1}, {2}}))
	if err != nil {
		t.Fatal(err)
	}
	sameNamedRight, err := New(Column("key", [][]int{{2}}))
	if err != nil {
		t.Fatal(err)
	}
	using, err := sameNamedLeft.InnerJoin(sameNamedRight, UsingBy[[]int]("key", intSliceHasher{}))
	if err != nil {
		t.Fatal(err)
	}
	if using.Len() != 1 {
		t.Fatalf("UsingBy result length = %d", using.Len())
	}
}

func TestCustomHasherJoinDoesNotHashNullKeys(t *testing.T) {
	value := 1
	leftKey := series.FromOptionals([]series.Optional[*int]{series.None[*int](), series.Some(&value)})
	rightKey := series.FromOptionals([]series.Optional[*int]{series.Some(&value), series.None[*int]()})
	left, err := New(Column("left", []int{0, 1}))
	if err != nil {
		t.Fatal(err)
	}
	right, err := New(Column("right", []int{0, 1}))
	if err != nil {
		t.Fatal(err)
	}
	key := OnBy(leftKey, rightKey, dereferencedIntHasher{})

	inner, err := left.InnerJoin(right, key)
	if err != nil {
		t.Fatal(err)
	}
	if inner.Len() != 1 {
		t.Fatalf("InnerJoin length = %d, want 1", inner.Len())
	}
	leftJoined, err := left.LeftJoin(right, key)
	if err != nil {
		t.Fatal(err)
	}
	if leftJoined.Len() != 2 {
		t.Fatalf("LeftJoin length = %d, want 2", leftJoined.Len())
	}
	rightJoined, err := left.RightJoin(right, key)
	if err != nil {
		t.Fatal(err)
	}
	if rightJoined.Len() != 2 {
		t.Fatalf("RightJoin length = %d, want 2", rightJoined.Len())
	}
	full, err := left.FullJoin(right, key)
	if err != nil {
		t.Fatal(err)
	}
	if full.Len() != 3 {
		t.Fatalf("FullJoin length = %d, want 3", full.Len())
	}
	semi, err := left.SemiJoin(right, key)
	if err != nil {
		t.Fatal(err)
	}
	if semi.Len() != 1 {
		t.Fatalf("SemiJoin length = %d, want 1", semi.Len())
	}
	anti, err := left.AntiJoin(right, key)
	if err != nil {
		t.Fatal(err)
	}
	if anti.Len() != 1 {
		t.Fatalf("AntiJoin length = %d, want 1", anti.Len())
	}
}

func TestOuterJoinsPreserveTypedColumnStorage(t *testing.T) {
	left, err := New(Column("key", []int{1, 2}), Column("left", []string{"a", "b"}))
	if err != nil {
		t.Fatal(err)
	}
	right, err := New(Column("key", []int{2, 3}), Column("right", []string{"x", "y"}))
	if err != nil {
		t.Fatal(err)
	}

	leftJoined, err := left.LeftJoin(right, Using[int]("key"))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := leftJoined.columns[0].(columnSpec[int]); !ok {
		t.Fatalf("LeftJoin key storage is %T, want columnSpec[int]", leftJoined.columns[0])
	}
	if _, ok := leftJoined.columns[2].(columnSpec[string]); !ok {
		t.Fatalf("LeftJoin nullable right storage is %T, want columnSpec[string]", leftJoined.columns[2])
	}

	full, err := left.FullJoin(right, Using[int]("key"))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := full.columns[0].(columnSpec[int]); !ok {
		t.Fatalf("FullJoin key storage is %T, want columnSpec[int]", full.columns[0])
	}
	if _, ok := full.columns[1].(columnSpec[string]); !ok {
		t.Fatalf("FullJoin nullable left storage is %T, want columnSpec[string]", full.columns[1])
	}
	if _, ok := full.columns[2].(columnSpec[string]); !ok {
		t.Fatalf("FullJoin nullable right storage is %T, want columnSpec[string]", full.columns[2])
	}
}

func TestRecordBackedOuterJoin(t *testing.T) {
	type leftRecord struct {
		Key  int
		Left string
	}
	type rightRecord struct {
		Key   int
		Right string
	}
	left, err := FromRecords([]leftRecord{{Key: 1, Left: "a"}, {Key: 2, Left: "b"}})
	if err != nil {
		t.Fatal(err)
	}
	right, err := FromRecords([]rightRecord{{Key: 2, Right: "x"}, {Key: 3, Right: "y"}})
	if err != nil {
		t.Fatal(err)
	}
	joined, err := left.FullJoin(right, Using[int]("Key"))
	if err != nil {
		t.Fatal(err)
	}
	keys, err := joined.Column[int]("Key")
	if err != nil {
		t.Fatal(err)
	}
	leftValues, err := joined.Column[string]("Left")
	if err != nil {
		t.Fatal(err)
	}
	rightValues, err := joined.Column[string]("Right")
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(keys.Values(), []int{1, 2, 3}) {
		t.Fatalf("keys = %v", keys.Values())
	}
	if got := leftValues.Optionals(); !reflect.DeepEqual(got, []series.Optional[string]{series.Some("a"), series.Some("b"), series.None[string]()}) {
		t.Fatalf("left values = %#v", got)
	}
	if got := rightValues.Optionals(); !reflect.DeepEqual(got, []series.Optional[string]{series.None[string](), series.Some("x"), series.Some("y")}) {
		t.Fatalf("right values = %#v", got)
	}
}

func TestJoinErrors(t *testing.T) {
	left, err := New(Column("key", []int{1}), Column("same", []string{"a"}))
	if err != nil {
		t.Fatal(err)
	}
	right, err := New(Column("key", []int{1}), Column("same", []string{"b"}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := left.InnerJoin(right, Using[int]("key")); !errors.Is(err, ErrColumnConflict) {
		t.Fatalf("conflict error = %v", err)
	}
	if _, err := left.InnerJoin(right, On(series.New([]int{}), series.New([]int{1}))); !errors.Is(err, ErrRowCount) {
		t.Fatalf("length error = %v", err)
	}
	if _, err := left.InnerJoin(right, Using[string]("key")); !errors.Is(err, ErrColumnType) {
		t.Fatalf("type error = %v", err)
	}
	if _, err := left.InnerJoin(right, Using[int]("missing")); !errors.Is(err, ErrColumnNotFound) {
		t.Fatalf("missing error = %v", err)
	}
	if _, err := left.InnerJoin(right, OnBy[int](series.New([]int{1}), series.New([]int{1}), nil)); !errors.Is(err, ErrUnsupported) {
		t.Fatalf("nil hasher error = %v", err)
	}
}

func TestCrossJoin(t *testing.T) {
	left, err := New(Column("left", []int{1, 2}))
	if err != nil {
		t.Fatal(err)
	}
	right, err := New(Column("right", []string{"a", "b", "c"}))
	if err != nil {
		t.Fatal(err)
	}
	joined, err := left.CrossJoin(right)
	if err != nil {
		t.Fatal(err)
	}
	leftValues, _ := joined.Column[int]("left")
	rightValues, _ := joined.Column[string]("right")
	if !slices.Equal(leftValues.Values(), []int{1, 1, 1, 2, 2, 2}) || !slices.Equal(rightValues.Values(), []string{"a", "b", "c", "a", "b", "c"}) {
		t.Fatalf("cross values = %v, %v", leftValues.Values(), rightValues.Values())
	}
	conflicting, err := New(Column("left", []int{1}))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := left.CrossJoin(conflicting); !errors.Is(err, ErrColumnConflict) {
		t.Fatalf("conflict error = %v", err)
	}
}

func TestCrossJoinRowCountOverflowReturnsError(t *testing.T) {
	maxInt := int(^uint(0) >> 1)
	_, err := (Frame{rowCount: maxInt}).CrossJoin(Frame{rowCount: 2})
	if !errors.Is(err, ErrRowCount) {
		t.Fatalf("CrossJoin() overflow error = %v, want ErrRowCount", err)
	}
}

func BenchmarkInnerJoin(b *testing.B) {
	const size = 10_000
	keys := make([]int, size)
	for i := range keys {
		keys[i] = i
	}
	left, err := New(Column("left", keys))
	if err != nil {
		b.Fatal(err)
	}
	right, err := New(Column("right", keys))
	if err != nil {
		b.Fatal(err)
	}
	joinKey := On(series.New(keys), series.New(keys))
	b.ReportAllocs()
	for b.Loop() {
		if _, err := left.InnerJoin(right, joinKey); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkLeftJoin(b *testing.B) {
	const size = 10_000
	keys := make([]int, size)
	for i := range keys {
		keys[i] = i
	}
	left, err := New(Column("left", keys))
	if err != nil {
		b.Fatal(err)
	}
	right, err := New(Column("right", keys))
	if err != nil {
		b.Fatal(err)
	}
	joinKey := On(series.New(keys), series.New(keys))
	b.ReportAllocs()
	for b.Loop() {
		if _, err := left.LeftJoin(right, joinKey); err != nil {
			b.Fatal(err)
		}
	}
}

func assertJoinColumns(t *testing.T, frame Frame, wantKeys []series.Optional[int], wantLeft, wantRight []series.Optional[string]) {
	t.Helper()
	keys, err := frame.Column[int]("key")
	if err != nil {
		t.Fatal(err)
	}
	left, err := frame.Column[string]("left")
	if err != nil {
		t.Fatal(err)
	}
	right, err := frame.Column[string]("right")
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(keys.Optionals(), wantKeys) || !reflect.DeepEqual(left.Optionals(), wantLeft) || !reflect.DeepEqual(right.Optionals(), wantRight) {
		t.Fatalf("joined columns:\nkeys  = %#v\nleft  = %#v\nright = %#v", keys.Optionals(), left.Optionals(), right.Optionals())
	}
}

type dereferencedIntHasher struct{}

func (dereferencedIntHasher) Hash(hash *maphash.Hash, value *int) {
	maphash.WriteComparable(hash, *value)
}

func (dereferencedIntHasher) Equal(left, right *int) bool {
	return *left == *right
}
