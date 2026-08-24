package dataframe

import (
	"errors"
	"hash/maphash"
	"math"
	"reflect"
	"runtime"
	"slices"
	"testing"

	"github.com/joeychilson/dataframe/internal/hashmap"
	"github.com/joeychilson/dataframe/series"
)

func TestOnJoin_PreservesLeftMajorMatchOrder(t *testing.T) {
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

func TestUsingJoins_CoalesceKeysAndApplyJoinSemantics(t *testing.T) {
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

func TestUsingInnerJoin_PreservesCoalescedKeyNullability(t *testing.T) {
	dense := series.New([]int{1})
	nullable, err := series.NewNullable([]int{1}, []bool{true})
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name        string
		left, right series.Series[int]
		nullable    bool
	}{
		{name: "keeps non-null inputs non-null", left: dense, right: dense},
		{name: "preserves a nullable left key", left: nullable, right: dense, nullable: true},
		{name: "widens for a nullable right key", left: dense, right: nullable, nullable: true},
		{name: "preserves two nullable keys", left: nullable, right: nullable, nullable: true},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			left, frameErr := New(ColumnFromSeries("key", test.left))
			if frameErr != nil {
				t.Fatal(frameErr)
			}
			right, frameErr := New(ColumnFromSeries("key", test.right))
			if frameErr != nil {
				t.Fatal(frameErr)
			}
			joined, joinErr := left.InnerJoin(right, Using[int]("key"))
			if joinErr != nil {
				t.Fatal(joinErr)
			}
			if got := joined.Schema()[0].Nullable; got != test.nullable {
				t.Fatalf("coalesced key nullable = %t, want %t", got, test.nullable)
			}
		})
	}
}

func TestUsingColumnsBy_JoinsDifferentlyNamedCustomKeys(t *testing.T) {
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

func TestSemiAndAntiJoin_SelectMatchedAndUnmatchedLeftRows(t *testing.T) {
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

func TestSemiAndAntiJoinIgnoreOutputColumnConflicts(t *testing.T) {
	left, err := New(Column("key", []int{1}), Column("same", []string{"left"}))
	if err != nil {
		t.Fatal(err)
	}
	right, err := New(Column("key", []int{1}), Column("same", []string{"right"}))
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
	if semi.Len() != 1 || anti.Len() != 0 || !slices.Equal(semi.Names(), left.Names()) || !slices.Equal(anti.Names(), left.Names()) {
		t.Fatalf("semi/anti shapes = %dx%d and %dx%d", semi.Len(), semi.Width(), anti.Len(), anti.Width())
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
	if _, ok := leftJoined.columns[0].values.(typedData[int]); !ok {
		t.Fatalf("LeftJoin key storage is %T, want typedData[int]", leftJoined.columns[0].values)
	}
	if _, ok := leftJoined.columns[2].values.(typedData[string]); !ok {
		t.Fatalf("LeftJoin nullable right storage is %T, want typedData[string]", leftJoined.columns[2].values)
	}

	full, err := left.FullJoin(right, Using[int]("key"))
	if err != nil {
		t.Fatal(err)
	}
	if _, ok := full.columns[0].values.(typedData[int]); !ok {
		t.Fatalf("FullJoin key storage is %T, want typedData[int]", full.columns[0].values)
	}
	if _, ok := full.columns[1].values.(typedData[string]); !ok {
		t.Fatalf("FullJoin nullable left storage is %T, want typedData[string]", full.columns[1].values)
	}
	if _, ok := full.columns[2].values.(typedData[string]); !ok {
		t.Fatalf("FullJoin nullable right storage is %T, want typedData[string]", full.columns[2].values)
	}
}

func TestRecordBackedOuterJoin_PreservesReflectedStorage(t *testing.T) {
	type value string
	type leftRecord struct {
		Key  int
		Left value
	}
	type rightRecord struct {
		Key   int
		Right value
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
	leftValues, err := joined.Column[value]("Left")
	if err != nil {
		t.Fatal(err)
	}
	rightValues, err := joined.Column[value]("Right")
	if err != nil {
		t.Fatal(err)
	}
	if !slices.Equal(keys.Values(), []int{1, 2, 3}) {
		t.Fatalf("keys = %v", keys.Values())
	}
	if got := leftValues.Optionals(); !reflect.DeepEqual(got, []series.Optional[value]{series.Some(value("a")), series.Some(value("b")), series.None[value]()}) {
		t.Fatalf("left values = %#v", got)
	}
	if got := rightValues.Optionals(); !reflect.DeepEqual(got, []series.Optional[value]{series.None[value](), series.Some(value("x")), series.Some(value("y"))}) {
		t.Fatalf("right values = %#v", got)
	}
}

func TestJoins_RejectInvalidKeysAndConflictingSchemas(t *testing.T) {
	left, err := New(Column("key", []int{1}), Column("same", []string{"a"}))
	if err != nil {
		t.Fatal(err)
	}
	right, err := New(Column("key", []int{1}), Column("same", []string{"b"}))
	if err != nil {
		t.Fatal(err)
	}
	if _, joinErr := left.InnerJoin(right, Using[int]("key")); !errors.Is(joinErr, ErrColumnConflict) {
		t.Fatalf("conflict error = %v", joinErr)
	}
	if _, joinErr := left.InnerJoin(right, On(series.New([]int{}), series.New([]int{1}))); !errors.Is(joinErr, ErrRowCount) {
		t.Fatalf("length error = %v", joinErr)
	}
	if _, joinErr := left.InnerJoin(right, Using[string]("key")); !errors.Is(joinErr, ErrColumnType) {
		t.Fatalf("type error = %v", joinErr)
	}
	if _, joinErr := left.InnerJoin(right, Using[int]("missing")); !errors.Is(joinErr, ErrColumnNotFound) {
		t.Fatalf("missing error = %v", joinErr)
	}
	if _, joinErr := left.InnerJoin(right, OnBy[int](series.New([]int{1}), series.New([]int{1}), nil)); !errors.Is(joinErr, ErrUnsupported) {
		t.Fatalf("nil hasher error = %v", joinErr)
	}
	if _, joinErr := left.InnerJoin(right, UsingBy[int]("key", nil)); !errors.Is(joinErr, ErrUnsupported) {
		t.Fatalf("stored nil hasher error = %v", joinErr)
	}
}

func TestCrossJoin_ProducesCartesianProduct(t *testing.T) {
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
	if _, joinErr := left.CrossJoin(conflicting); !errors.Is(joinErr, ErrColumnConflict) {
		t.Fatalf("conflict error = %v", joinErr)
	}

	zeroLeft := Frame{rowCount: left.Len()}
	zeroRight := Frame{rowCount: right.Len()}
	for _, test := range []struct {
		name        string
		left, right Frame
		width       int
	}{
		{name: "both zero-width frames retain product rows", left: zeroLeft, right: zeroRight},
		{name: "a zero-width right frame retains left columns", left: left, right: zeroRight, width: 1},
		{name: "a zero-width left frame retains right columns", left: zeroLeft, right: right, width: 1},
	} {
		crossJoined, joinErr := test.left.CrossJoin(test.right)
		if joinErr != nil {
			t.Fatal(joinErr)
		}
		if crossJoined.Len() != 6 || crossJoined.Width() != test.width {
			t.Fatalf("%s zero-width CrossJoin shape = %dx%d", test.name, crossJoined.Len(), crossJoined.Width())
		}
	}
}

func TestZeroWidthOuterJoins_RetainExpectedRowCounts(t *testing.T) {
	leftKeys := series.New([]int{1, 2})
	rightKeys := series.New([]int{3, 4})
	zero := Frame{rowCount: 2}
	full, err := zero.FullJoin(zero, On(leftKeys, rightKeys))
	if err != nil {
		t.Fatal(err)
	}
	if full.Len() != 4 || full.Width() != 0 {
		t.Fatalf("zero-width FullJoin shape = %dx%d", full.Len(), full.Width())
	}
}

func TestCrossJoinRowCountOverflowReturnsError(t *testing.T) {
	_, err := (Frame{rowCount: math.MaxInt}).CrossJoin(Frame{rowCount: 2})
	if !errors.Is(err, ErrRowCount) {
		t.Fatalf("CrossJoin() overflow error = %v, want ErrRowCount", err)
	}
}

func TestPairedMatchingRowsPreallocateDuplicateResults(t *testing.T) {
	probe := series.New([]int{1, 1, 2})
	indexed := series.New([]int{1, 1, 2, 2})
	probeRows, indexedRows, err := pairedMatchingRows(probe, indexed, make(comparableJoinLookup[int], indexed.Len()), false)
	if err != nil {
		t.Fatal(err)
	}
	if want := []int{0, 0, 1, 1, 2, 2}; !slices.Equal(probeRows, want) {
		t.Fatalf("probe rows = %v, want %v", probeRows, want)
	}
	if want := []int{0, 1, 0, 1, 2, 3}; !slices.Equal(indexedRows, want) {
		t.Fatalf("indexed rows = %v, want %v", indexedRows, want)
	}
	if cap(probeRows) != len(probeRows) || cap(indexedRows) != len(indexedRows) {
		t.Fatalf("result capacities = %d and %d, want %d", cap(probeRows), cap(indexedRows), len(probeRows))
	}
}

func TestJoinIndexAllocatesDuplicateChainsLazily(t *testing.T) {
	unique := series.New([]int{1, 2, 3})
	uniqueIndex := newJoinIndex(unique, make(comparableJoinLookup[int], unique.Len()))
	if uniqueIndex.next != nil {
		t.Fatal("unique join index allocated a duplicate chain")
	}
	if row, found := uniqueIndex.lookup.Get(2); !found || row != 1 {
		t.Fatalf("unique match = (%d, %t), want (1, true)", row, found)
	}

	duplicate := series.New([]int{1, 2, 1})
	duplicateIndex := newJoinIndex(duplicate, make(comparableJoinLookup[int], duplicate.Len()))
	if len(duplicateIndex.next) != duplicate.Len() || len(duplicateIndex.duplicates) != 1 {
		t.Fatalf("duplicate storage = %d chain rows and %d matches, want %d and 1", len(duplicateIndex.next), len(duplicateIndex.duplicates), duplicate.Len())
	}
	matches, found := duplicateIndex.matches(duplicate, 0)
	if !found || matches.first != 0 || matches.count != 2 || duplicateIndex.next[0] != 2 {
		t.Fatalf("duplicate match = %+v, found %t, next %v", matches, found, duplicateIndex.next)
	}
}

func FuzzJoinsAgainstNestedLoop(f *testing.F) {
	f.Add([]byte{1, 3, 5, 5, 2}, []byte{5, 1, 7, 2})
	f.Add([]byte{}, []byte{})
	f.Fuzz(func(t *testing.T, leftData, rightData []byte) {
		decode := func(data []byte) []series.Optional[float64] {
			data = data[:min(len(data), 8)]
			keys := make([]series.Optional[float64], len(data))
			for i, value := range data {
				if value&1 == 0 {
					continue
				}
				var key float64
				switch (value >> 1) % 5 {
				case 0:
					key = math.NaN()
				case 1:
					key = 0
				case 2:
					key = 1
				case 3:
					key = -1
				default:
					key = float64(int8(value)) / 4
				}
				keys[i] = series.Some(key)
			}
			return keys
		}

		left := decode(leftData)
		right := decode(rightData)
		assertJoinModel(t, left, right, On(series.FromOptionals(left), series.FromOptionals(right)), func(leftKey, rightKey float64) bool {
			return leftKey == rightKey
		})
		hasher := fuzzFloatHasher{}
		assertJoinModel(t, left, right, OnBy(series.FromOptionals(left), series.FromOptionals(right), hasher), hasher.Equal)
	})
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

type joinPair struct {
	left  int
	right int
}

func assertJoinModel(t *testing.T, leftKeys, rightKeys []series.Optional[float64], key JoinKey[float64], equal func(float64, float64) bool) {
	t.Helper()
	leftRows := make([]int, len(leftKeys))
	for i := range leftRows {
		leftRows[i] = i
	}
	rightRows := make([]int, len(rightKeys))
	for i := range rightRows {
		rightRows[i] = i
	}
	left, err := New(Column("left", leftRows))
	if err != nil {
		t.Fatal(err)
	}
	right, err := New(Column("right", rightRows))
	if err != nil {
		t.Fatal(err)
	}
	operations := []struct {
		name string
		kind joinKind
		join func(Frame, Frame, JoinKey[float64]) (Frame, error)
	}{
		{name: "InnerJoin matches the reference model", kind: innerJoin, join: func(leftFrame, rightFrame Frame, joinKey JoinKey[float64]) (Frame, error) {
			return leftFrame.InnerJoin(rightFrame, joinKey)
		}},
		{name: "LeftJoin matches the reference model", kind: leftJoin, join: func(leftFrame, rightFrame Frame, joinKey JoinKey[float64]) (Frame, error) {
			return leftFrame.LeftJoin(rightFrame, joinKey)
		}},
		{name: "RightJoin matches the reference model", kind: rightJoin, join: func(leftFrame, rightFrame Frame, joinKey JoinKey[float64]) (Frame, error) {
			return leftFrame.RightJoin(rightFrame, joinKey)
		}},
		{name: "FullJoin matches the reference model", kind: fullJoin, join: func(leftFrame, rightFrame Frame, joinKey JoinKey[float64]) (Frame, error) {
			return leftFrame.FullJoin(rightFrame, joinKey)
		}},
		{name: "SemiJoin matches the reference model", kind: semiJoin, join: func(leftFrame, rightFrame Frame, joinKey JoinKey[float64]) (Frame, error) {
			return leftFrame.SemiJoin(rightFrame, joinKey)
		}},
		{name: "AntiJoin matches the reference model", kind: antiJoin, join: func(leftFrame, rightFrame Frame, joinKey JoinKey[float64]) (Frame, error) {
			return leftFrame.AntiJoin(rightFrame, joinKey)
		}},
	}
	for _, operation := range operations {
		joined, joinErr := operation.join(left, right, key)
		if joinErr != nil {
			t.Fatalf("%s: %v", operation.name, joinErr)
		}
		want := referenceJoinPairs(leftKeys, rightKeys, operation.kind, equal)
		leftColumn, columnErr := joined.Column[int]("left")
		if columnErr != nil {
			t.Fatalf("%s left column: %v", operation.name, columnErr)
		}
		if operation.kind == semiJoin || operation.kind == antiJoin {
			got := leftColumn.Values()
			wantRows := make([]int, len(want))
			for i, pair := range want {
				wantRows[i] = pair.left
			}
			if !slices.Equal(got, wantRows) {
				t.Fatalf("%s rows = %v, want %v", operation.name, got, wantRows)
			}
			continue
		}
		rightColumn, columnErr := joined.Column[int]("right")
		if columnErr != nil {
			t.Fatalf("%s right column: %v", operation.name, columnErr)
		}
		got := make([]joinPair, joined.Len())
		for i := range got {
			got[i].left, _ = leftColumn.At(i)
			got[i].right, _ = rightColumn.At(i)
			if !leftColumn.IsValid(i) {
				got[i].left = -1
			}
			if !rightColumn.IsValid(i) {
				got[i].right = -1
			}
		}
		if !slices.Equal(got, want) {
			t.Fatalf("%s pairs = %v, want %v", operation.name, got, want)
		}
	}
}

func referenceJoinPairs(left, right []series.Optional[float64], kind joinKind, equal func(float64, float64) bool) []joinPair {
	switch kind {
	case innerJoin, leftJoin, rightJoin, fullJoin, semiJoin, antiJoin:
	default:
		panic("invalid join kind")
	}
	matches := func(leftRow, rightRow int) bool {
		return left[leftRow].Valid && right[rightRow].Valid && equal(left[leftRow].Value, right[rightRow].Value)
	}
	var pairs []joinPair
	if kind == rightJoin {
		for rightRow := range right {
			matched := false
			for leftRow := range left {
				if matches(leftRow, rightRow) {
					pairs = append(pairs, joinPair{left: leftRow, right: rightRow})
					matched = true
				}
			}
			if !matched {
				pairs = append(pairs, joinPair{left: -1, right: rightRow})
			}
		}
		return pairs
	}

	matchedRight := make([]bool, len(right))
	for leftRow := range left {
		matched := false
		for rightRow := range right {
			if !matches(leftRow, rightRow) {
				continue
			}
			matched = true
			matchedRight[rightRow] = true
			if kind == innerJoin || kind == leftJoin || kind == fullJoin {
				pairs = append(pairs, joinPair{left: leftRow, right: rightRow})
			}
		}
		switch {
		case kind == semiJoin && matched, kind == antiJoin && !matched:
			pairs = append(pairs, joinPair{left: leftRow, right: -1})
		case (kind == leftJoin || kind == fullJoin) && !matched:
			pairs = append(pairs, joinPair{left: leftRow, right: -1})
		}
	}
	if kind == fullJoin {
		for rightRow, matched := range matchedRight {
			if !matched {
				pairs = append(pairs, joinPair{left: -1, right: rightRow})
			}
		}
	}
	return pairs
}

type fuzzFloatHasher struct{}

func (fuzzFloatHasher) Hash(hash *maphash.Hash, value float64) {
	bits := math.Float64bits(value)
	if math.IsNaN(value) {
		bits = math.Float64bits(math.NaN())
	} else if value == 0 {
		bits = 0
	}
	maphash.WriteComparable(hash, bits)
}

func (fuzzFloatHasher) Equal(left, right float64) bool {
	return left == right || math.IsNaN(left) && math.IsNaN(right)
}

type dereferencedIntHasher struct{}

func (dereferencedIntHasher) Hash(hash *maphash.Hash, value *int) {
	maphash.WriteComparable(hash, *value)
}

func (dereferencedIntHasher) Equal(left, right *int) bool {
	return *left == *right
}

func BenchmarkJoins(b *testing.B) {
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
	key := On(series.New(keys), series.New(keys))
	operations := []struct {
		name string
		join func(Frame, Frame, JoinKey[int]) (Frame, error)
	}{
		{name: "inner", join: func(leftFrame, rightFrame Frame, joinKey JoinKey[int]) (Frame, error) {
			return leftFrame.InnerJoin(rightFrame, joinKey)
		}},
		{name: "left", join: func(leftFrame, rightFrame Frame, joinKey JoinKey[int]) (Frame, error) {
			return leftFrame.LeftJoin(rightFrame, joinKey)
		}},
		{name: "full", join: func(leftFrame, rightFrame Frame, joinKey JoinKey[int]) (Frame, error) {
			return leftFrame.FullJoin(rightFrame, joinKey)
		}},
		{name: "semi", join: func(leftFrame, rightFrame Frame, joinKey JoinKey[int]) (Frame, error) {
			return leftFrame.SemiJoin(rightFrame, joinKey)
		}},
	}
	for _, operation := range operations {
		b.Run("one-to-one/"+operation.name, func(b *testing.B) {
			b.ReportAllocs()
			var result Frame
			for b.Loop() {
				var joinErr error
				result, joinErr = operation.join(left, right, key)
				if joinErr != nil {
					b.Fatal(joinErr)
				}
			}
			runtime.KeepAlive(result)
		})
	}

	manyKeys := make([]int, size)
	for i := range manyKeys {
		manyKeys[i] = i % 1_000
	}
	manyLeft, err := New(Column("left", manyKeys))
	if err != nil {
		b.Fatal(err)
	}
	manyRight, err := New(Column("right", manyKeys))
	if err != nil {
		b.Fatal(err)
	}
	manyKey := On(series.New(manyKeys), series.New(manyKeys))
	b.Run("many-to-many/inner", func(b *testing.B) {
		b.ReportAllocs()
		var result Frame
		for b.Loop() {
			var joinErr error
			result, joinErr = manyLeft.InnerJoin(manyRight, manyKey)
			if joinErr != nil {
				b.Fatal(joinErr)
			}
		}
		runtime.KeepAlive(result)
	})

	const crossSize = 256
	crossLeft, err := New(Column("left", make([]int, crossSize)))
	if err != nil {
		b.Fatal(err)
	}
	crossRight, err := New(Column("right", make([]int, crossSize)))
	if err != nil {
		b.Fatal(err)
	}
	b.Run("cross", func(b *testing.B) {
		b.ReportAllocs()
		var result Frame
		for b.Loop() {
			var joinErr error
			result, joinErr = crossLeft.CrossJoin(crossRight)
			if joinErr != nil {
				b.Fatal(joinErr)
			}
		}
		runtime.KeepAlive(result)
	})
}

func BenchmarkJoinIndex(b *testing.B) {
	const size = 10_000
	reference := func(probe, indexed series.Series[int]) ([]int, []int) {
		matchesByKey := make(map[int][]int, indexed.Len())
		for row, value := range indexed.Present() {
			matchesByKey[value] = append(matchesByKey[value], row)
		}
		probeRows := make([]int, 0, probe.Len())
		indexedRows := make([]int, 0, probe.Len())
		for probeRow, value := range probe.Present() {
			for _, indexedRow := range matchesByKey[value] {
				probeRows = append(probeRows, probeRow)
				indexedRows = append(indexedRows, indexedRow)
			}
		}
		return probeRows, indexedRows
	}
	inputs := []struct {
		name    string
		probe   series.Series[int]
		indexed series.Series[int]
	}{
		{name: "OneToOne", probe: series.NewFunc(size, func(i int) int { return i }), indexed: series.NewFunc(size, func(i int) int { return i })},
		{name: "SparseDuplicates", probe: series.NewFunc(size, func(i int) int { return i / 2 }), indexed: series.NewFunc(size, func(i int) int { return i / 2 })},
		{name: "HeavyDuplicates", probe: series.NewFunc(size, func(i int) int { return i % 1_000 }), indexed: series.NewFunc(size, func(i int) int { return i % 1_000 })},
		{name: "NullKeys", probe: series.NewNullableFunc(size, func(i int) (int, bool) { return i % 1_000, i%4 != 0 }), indexed: series.NewNullableFunc(size, func(i int) (int, bool) { return i % 1_000, i%4 != 0 })},
	}
	for _, input := range inputs {
		wantProbe, wantIndexed := reference(input.probe, input.indexed)
		gotProbe, gotIndexed, matchErr := pairedMatchingRows(input.probe, input.indexed, make(comparableJoinLookup[int], input.indexed.Len()), false)
		if matchErr != nil {
			b.Fatal(matchErr)
		}
		if !slices.Equal(gotProbe, wantProbe) || !slices.Equal(gotIndexed, wantIndexed) {
			b.Fatalf("%s reference mismatch", input.name)
		}
		b.Run(input.name, func(b *testing.B) {
			b.Run("Optimized", func(b *testing.B) {
				b.ReportAllocs()
				var probeRows, indexedRows []int
				for b.Loop() {
					var benchmarkErr error
					probeRows, indexedRows, benchmarkErr = pairedMatchingRows(input.probe, input.indexed, make(comparableJoinLookup[int], input.indexed.Len()), false)
					if benchmarkErr != nil {
						b.Fatal(benchmarkErr)
					}
				}
				runtime.KeepAlive(probeRows)
				runtime.KeepAlive(indexedRows)
			})
			b.Run("Reference", func(b *testing.B) {
				b.ReportAllocs()
				var probeRows, indexedRows []int
				for b.Loop() {
					probeRows, indexedRows = reference(input.probe, input.indexed)
				}
				runtime.KeepAlive(probeRows)
				runtime.KeepAlive(indexedRows)
			})
		})
	}
	customValues := make([]float64, size)
	for i := range customValues {
		customValues[i] = float64(i % 1_000)
		if i%1_000 == 0 {
			customValues[i] = math.NaN()
		}
	}
	customKeys := series.New(customValues)
	b.Run("CustomHasher", func(b *testing.B) {
		b.Run("Optimized", func(b *testing.B) {
			b.ReportAllocs()
			var probeRows, indexedRows []int
			for b.Loop() {
				lookup := hashmap.New[float64, int](fuzzFloatHasher{}, customKeys.Len())
				var matchErr error
				probeRows, indexedRows, matchErr = pairedMatchingRows(customKeys, customKeys, lookup, false)
				if matchErr != nil {
					b.Fatal(matchErr)
				}
			}
			runtime.KeepAlive(probeRows)
			runtime.KeepAlive(indexedRows)
		})
		b.Run("Reference", func(b *testing.B) {
			b.ReportAllocs()
			var probeRows, indexedRows []int
			for b.Loop() {
				matchesByKey := hashmap.New[float64, []int](fuzzFloatHasher{}, customKeys.Len())
				for row, value := range customKeys.Present() {
					matches, _ := matchesByKey.Get(value)
					matchesByKey.Set(value, append(matches, row))
				}
				probeRows = nil
				indexedRows = nil
				for probeRow, value := range customKeys.Present() {
					matches, _ := matchesByKey.Get(value)
					for _, indexedRow := range matches {
						probeRows = append(probeRows, probeRow)
						indexedRows = append(indexedRows, indexedRow)
					}
				}
			}
			runtime.KeepAlive(probeRows)
			runtime.KeepAlive(indexedRows)
		})
	})
}
