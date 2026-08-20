package series_test

import (
	"errors"
	"math"
	"slices"
	"strconv"
	"testing"

	"github.com/joeychilson/dataframe/series"
)

type score int

func TestNewCopiesInput(t *testing.T) {
	input := []int{1, 2}
	numbers := series.New(input)
	input[0] = 99

	if got, want := numbers.Values(), []int{1, 2}; !slices.Equal(got, want) {
		t.Fatalf("values = %v, want %v", got, want)
	}
}

func TestConcatPreservesOrderAndWidensNullability(t *testing.T) {
	middle, err := series.NewNullable(
		[]score{99, 3},
		[]bool{false, true},
	)
	if err != nil {
		t.Fatal(err)
	}

	combined := series.New([]score{1, 2}).Concat(
		middle,
		series.New([]score{4}),
	)
	if got, want := combined.Values(), []score{1, 2, 99, 3, 4}; !slices.Equal(got, want) {
		t.Fatalf("values = %v, want %v", got, want)
	}
	if got, want := combined.Validity(), []bool{true, true, false, true, true}; !slices.Equal(got, want) {
		t.Fatalf("validity = %v, want %v", got, want)
	}
	if !combined.Nullable() {
		t.Fatal("combined Series is not nullable")
	}
	if got, want := middle.Validity(), []bool{false, true}; !slices.Equal(got, want) {
		t.Fatalf("input validity = %v, want %v", got, want)
	}
}

func TestConcatNonNullableInputsStayNonNullable(t *testing.T) {
	combined := series.New([]int{1, 2}).Concat(series.New([]int{3}))
	if combined.Nullable() {
		t.Fatal("non-nullable inputs produced a nullable Series")
	}
	if got, want := combined.Values(), []int{1, 2, 3}; !slices.Equal(got, want) {
		t.Fatalf("values = %v, want %v", got, want)
	}
}

func TestConcatHandlesSingleAndEmptyNullableInputs(t *testing.T) {
	one, err := series.NewNullable([]int{2}, []bool{true})
	if err != nil {
		t.Fatal(err)
	}
	unchanged := one.Concat()
	if got, valid := unchanged.At(0); !valid || got != 2 || !unchanged.Nullable() {
		t.Fatalf("single input = %v, %v, nullable %v; want 2, true, true", got, valid, unchanged.Nullable())
	}

	emptyNullable, err := series.NewNullable([]int{}, []bool{})
	if err != nil {
		t.Fatal(err)
	}
	widened := series.New([]int{1, 2}).Concat(emptyNullable)
	if !widened.Nullable() {
		t.Fatal("empty nullable input did not widen result schema")
	}
	if got, want := widened.Validity(), []bool{true, true}; !slices.Equal(got, want) {
		t.Fatalf("widened validity = %v, want %v", got, want)
	}
}

func TestGenericMethodsPreserveNulls(t *testing.T) {
	numbers, err := series.NewNullable(
		[]int{2, 999, 4},
		[]bool{true, false, true},
	)
	if err != nil {
		t.Fatal(err)
	}

	calls := 0
	strings := numbers.Map(func(value int) string {
		calls++
		return strconv.Itoa(value)
	})
	if calls != 2 {
		t.Fatalf("mapper called %d times, want 2", calls)
	}
	if got, want := strings.Validity(), []bool{true, false, true}; !slices.Equal(got, want) {
		t.Fatalf("validity = %v, want %v", got, want)
	}
	if got := strings.Reduce("", func(acc, value string) string { return acc + value }); got != "24" {
		t.Fatalf("reduce = %q, want %q", got, "24")
	}
}

func TestPresentSkipsNullsAndStopsEarly(t *testing.T) {
	numbers, err := series.NewNullable(
		[]int{2, 99, 4},
		[]bool{true, false, true},
	)
	if err != nil {
		t.Fatal(err)
	}

	if got, want := slices.Collect(numbers.Present()), []int{2, 4}; !slices.Equal(got, want) {
		t.Fatalf("present values = %v, want %v", got, want)
	}

	visited := 0
	for range numbers.Present() {
		visited++
		break
	}
	if visited != 1 {
		t.Fatalf("visited = %d, want 1", visited)
	}
}

func TestAllYieldsValueAndValidity(t *testing.T) {
	numbers, err := series.NewNullable(
		[]int{2, 99, 4},
		[]bool{true, false, true},
	)
	if err != nil {
		t.Fatal(err)
	}

	var values []int
	var valids []bool
	for value, valid := range numbers.All() {
		values = append(values, value)
		valids = append(valids, valid)
	}
	if got, want := valids, []bool{true, false, true}; !slices.Equal(got, want) {
		t.Fatalf("valids = %v, want %v", got, want)
	}
	if values[0] != 2 || values[2] != 4 {
		t.Fatalf("present values = %d, %d; want 2, 4", values[0], values[2])
	}

	dense := series.New([]int{1, 2})
	var denseValids []bool
	for _, valid := range dense.All() {
		denseValids = append(denseValids, valid)
	}
	if got, want := denseValids, []bool{true, true}; !slices.Equal(got, want) {
		t.Fatalf("dense valids = %v, want %v", got, want)
	}

	visited := 0
	for range numbers.All() {
		visited++
		break
	}
	if visited != 1 {
		t.Fatalf("visited = %d, want 1", visited)
	}
}

func TestEachYieldsRowIndexesAndSkipsNulls(t *testing.T) {
	numbers, err := series.NewNullable(
		[]int{2, 99, 4},
		[]bool{true, false, true},
	)
	if err != nil {
		t.Fatal(err)
	}

	type row struct {
		index int
		value int
	}

	var rows []row
	for i, value := range numbers.Each() {
		rows = append(rows, row{i, value})
	}
	if got, want := rows, []row{{0, 2}, {2, 4}}; !slices.Equal(got, want) {
		t.Fatalf("rows = %v, want %v", got, want)
	}

	var dense []row
	for i, value := range series.New([]int{5, 6}).Each() {
		dense = append(dense, row{i, value})
	}
	if got, want := dense, []row{{0, 5}, {1, 6}}; !slices.Equal(got, want) {
		t.Fatalf("dense rows = %v, want %v", got, want)
	}

	visited := 0
	for range numbers.Each() {
		visited++
		break
	}
	if visited != 1 {
		t.Fatalf("visited = %d, want 1", visited)
	}
}

func TestTryMapStopsAtErrorAndPreservesNulls(t *testing.T) {
	numbers, err := series.NewNullable(
		[]int{2, 99, 4},
		[]bool{true, false, true},
	)
	if err != nil {
		t.Fatal(err)
	}

	mapped, err := numbers.TryMap(func(value int) (string, error) {
		return strconv.Itoa(value), nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := mapped.Validity(), []bool{true, false, true}; !slices.Equal(got, want) {
		t.Fatalf("validity = %v, want %v", got, want)
	}
	if got, valid := mapped.At(0); !valid || got != "2" {
		t.Fatalf("row 0 = %q, %v; want 2, true", got, valid)
	}
	if got, valid := mapped.At(2); !valid || got != "4" {
		t.Fatalf("row 2 = %q, %v; want 4, true", got, valid)
	}

	_, err = numbers.TryMap(func(value int) (string, error) {
		if value == 4 {
			return "", errors.New("boom")
		}
		return strconv.Itoa(value), nil
	})
	if err == nil {
		t.Fatal("expected error")
	}
	if got := err.Error(); got != "map row 2: boom" {
		t.Fatalf("error = %q, want map row 2: boom", got)
	}

	dense := series.New([]int{2, 4})
	_, err = dense.TryMap(func(value int) (string, error) {
		if value == 4 {
			return "", errors.New("dense boom")
		}
		return strconv.Itoa(value), nil
	})
	if err == nil {
		t.Fatal("expected dense error")
	}
	if got := err.Error(); got != "map row 1: dense boom" {
		t.Fatalf("dense error = %q, want map row 1: dense boom", got)
	}
}

func TestMap2CombinesValuesAndPropagatesNulls(t *testing.T) {
	left, err := series.NewNullable(
		[]int{2, 99, 4, 6},
		[]bool{true, false, true, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	right, err := series.NewNullable(
		[]string{"a", "b", "c", "d"},
		[]bool{true, true, false, true},
	)
	if err != nil {
		t.Fatal(err)
	}

	calls := 0
	mapped := left.Map2(right, func(number int, text string) string {
		calls++
		return text + strconv.Itoa(number)
	})
	if calls != 2 {
		t.Fatalf("callback calls = %d, want 2", calls)
	}
	if got, want := mapped.Validity(), []bool{true, false, false, true}; !slices.Equal(got, want) {
		t.Fatalf("validity = %v, want %v", got, want)
	}
	if got, valid := mapped.At(0); !valid || got != "a2" {
		t.Fatalf("row 0 = (%q, %v), want (a2, true)", got, valid)
	}
	if got, valid := mapped.At(3); !valid || got != "d6" {
		t.Fatalf("row 3 = (%q, %v), want (d6, true)", got, valid)
	}

	dense := series.New([]int{1, 2}).Map2(
		series.New([]string{"x", "y"}),
		func(number int, text string) string { return text + strconv.Itoa(number) },
	)
	if got, want := dense.Values(), []string{"x1", "y2"}; !slices.Equal(got, want) {
		t.Fatalf("dense values = %v, want %v", got, want)
	}
	if dense.Nullable() {
		t.Fatal("dense Map2 result is nullable")
	}

	allPresent, err := series.NewNullable([]int{1}, []bool{true})
	if err != nil {
		t.Fatal(err)
	}
	if result := allPresent.Map2(series.New([]int{2}), func(a, b int) int { return a + b }); !result.Nullable() {
		t.Fatal("Map2 lost nullable schema with no current nulls")
	}
	if result := series.New([]int{2}).Map2(allPresent, func(a, b int) int { return a + b }); !result.Nullable() {
		t.Fatal("Map2 lost right-side nullable schema with no current nulls")
	}
}

func TestTryMap2StopsAtErrorAndPreservesNulls(t *testing.T) {
	left, err := series.NewNullable(
		[]int{2, 99, 4},
		[]bool{true, false, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	right := series.New([]int{10, 20, 30})

	mapped, err := left.TryMap2(right, func(a, b int) (int, error) {
		return a + b, nil
	})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := mapped.Validity(), []bool{true, false, true}; !slices.Equal(got, want) {
		t.Fatalf("validity = %v, want %v", got, want)
	}
	if got, valid := mapped.At(2); !valid || got != 34 {
		t.Fatalf("row 2 = (%d, %v), want (34, true)", got, valid)
	}

	boom := errors.New("boom")
	calls := 0
	_, err = left.TryMap2(right, func(a, b int) (int, error) {
		calls++
		if a == 4 {
			return 0, boom
		}
		return a + b, nil
	})
	if !errors.Is(err, boom) {
		t.Fatalf("error = %v, want wrapped boom", err)
	}
	if got, want := err.Error(), "map row 2: boom"; got != want {
		t.Fatalf("error = %q, want %q", got, want)
	}
	if calls != 2 {
		t.Fatalf("callback calls = %d, want 2", calls)
	}

	_, err = series.New([]int{1, 2}).TryMap2(
		series.New([]int{10, 20}),
		func(a, b int) (int, error) {
			if a == 2 {
				return 0, boom
			}
			return a + b, nil
		},
	)
	if got, want := err.Error(), "map row 1: boom"; got != want {
		t.Fatalf("dense error = %q, want %q", got, want)
	}
}

func TestMap2AndTryMap2PanicForDifferentLengths(t *testing.T) {
	tests := map[string]func(){
		"Map2": func() {
			series.New([]int{1}).Map2(series.New([]int{1, 2}), func(a, b int) int { return a + b })
		},
		"TryMap2": func() {
			series.New([]int{1}).TryMap2(series.New([]int{1, 2}), func(a, b int) (int, error) {
				return a + b, nil
			})
		},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("different lengths did not panic")
				}
			}()
			test()
		})
	}
}

func TestAggregationsSkipNullsAndPreserveNamedTypes(t *testing.T) {
	numbers, err := series.NewNullable(
		[]score{2, 99, 3},
		[]bool{true, false, true},
	)
	if err != nil {
		t.Fatal(err)
	}

	if got := numbers.Count(); got != 2 {
		t.Fatalf("count = %d, want 2", got)
	}
	if got, ok := series.Sum(numbers); !ok || got != score(5) {
		t.Fatalf("sum = %v, %v; want 5, true", got, ok)
	}
	if got, ok := series.Mean(numbers); !ok || got != 2.5 {
		t.Fatalf("mean = %v, %v; want 2.5, true", got, ok)
	}
	if got, ok := series.Min(numbers); !ok || got != score(2) {
		t.Fatalf("min = %v, %v; want 2, true", got, ok)
	}
	if got, ok := series.Max(numbers); !ok || got != score(3) {
		t.Fatalf("max = %v, %v; want 3, true", got, ok)
	}
}

func TestNumericAggregationsSupportFloat32AndInt64(t *testing.T) {
	floats := series.New([]float32{1.5, 2.5})
	if got, ok := series.Sum(floats); !ok || got != float32(4) {
		t.Fatalf("float32 sum = %v, %v; want 4, true", got, ok)
	}
	if got, ok := series.Mean(floats); !ok || got != 2 {
		t.Fatalf("float32 mean = %v, %v; want 2, true", got, ok)
	}

	integers := series.New([]int64{2, 3})
	if got, ok := series.Sum(integers); !ok || got != int64(5) {
		t.Fatalf("int64 sum = %v, %v; want 5, true", got, ok)
	}
}

func TestOrderedAggregationsSupportStrings(t *testing.T) {
	words := series.New([]string{"pear", "apple", "orange"})

	if got, ok := series.Min(words); !ok || got != "apple" {
		t.Fatalf("min = %q, %v; want apple, true", got, ok)
	}
	if got, ok := series.Max(words); !ok || got != "pear" {
		t.Fatalf("max = %q, %v; want pear, true", got, ok)
	}
}

func TestAggregationsReportNoPresentValues(t *testing.T) {
	empty := series.New([]int{})
	allNull, err := series.NewNullable(
		[]int{1, 2},
		[]bool{false, false},
	)
	if err != nil {
		t.Fatal(err)
	}

	for name, values := range map[string]series.Series[int]{
		"empty":    empty,
		"all null": allNull,
	} {
		t.Run(name, func(t *testing.T) {
			if got := values.Count(); got != 0 {
				t.Fatalf("count = %d, want 0", got)
			}
			if got, ok := series.Sum(values); ok || got != 0 {
				t.Fatalf("sum = %v, %v; want 0, false", got, ok)
			}
			if got, ok := series.Mean(values); ok || got != 0 {
				t.Fatalf("mean = %v, %v; want 0, false", got, ok)
			}
			if got, ok := series.Min(values); ok || got != 0 {
				t.Fatalf("min = %v, %v; want 0, false", got, ok)
			}
			if got, ok := series.Max(values); ok || got != 0 {
				t.Fatalf("max = %v, %v; want 0, false", got, ok)
			}
		})
	}
}

func TestMinAndMaxPropagatePresentNaN(t *testing.T) {
	numbers := series.New([]float64{2, math.NaN(), 1})

	minimum, ok := series.Min(numbers)
	if !ok || !math.IsNaN(minimum) {
		t.Fatalf("min = %v, %v; want NaN, true", minimum, ok)
	}
	maximum, ok := series.Max(numbers)
	if !ok || !math.IsNaN(maximum) {
		t.Fatalf("max = %v, %v; want NaN, true", maximum, ok)
	}
}

func TestAggregationsIgnoreNaNAtNullRows(t *testing.T) {
	numbers, err := series.NewNullable(
		[]float64{2, math.NaN(), 1},
		[]bool{true, false, true},
	)
	if err != nil {
		t.Fatal(err)
	}

	if got, ok := series.Min(numbers); !ok || got != 1 {
		t.Fatalf("min = %v, %v; want 1, true", got, ok)
	}
	if got, ok := series.Max(numbers); !ok || got != 2 {
		t.Fatalf("max = %v, %v; want 2, true", got, ok)
	}
}

func TestTakePreservesOrderAndNullability(t *testing.T) {
	numbers, err := series.NewNullable(
		[]int{2, 3, 4},
		[]bool{true, false, true},
	)
	if err != nil {
		t.Fatal(err)
	}

	taken := numbers.Take([]int{2, 0})
	if got, want := taken.Values(), []int{4, 2}; !slices.Equal(got, want) {
		t.Fatalf("values = %v, want %v", got, want)
	}
	if got, want := taken.Validity(), []bool{true, true}; !slices.Equal(got, want) {
		t.Fatalf("validity = %v, want %v", got, want)
	}
	if !taken.Nullable() {
		t.Fatal("Take removed nullable schema")
	}
}

func TestNewNullableRejectsMismatchedValidity(t *testing.T) {
	_, err := series.NewNullable([]int{1, 2}, []bool{true})
	if !errors.Is(err, series.ErrInvalidValidity) {
		t.Fatalf("error = %v, want ErrInvalidValidity", err)
	}
}

func TestNewNullableRequiresValidity(t *testing.T) {
	_, err := series.NewNullable([]int{1, 2}, nil)
	if !errors.Is(err, series.ErrValidityRequired) {
		t.Fatalf("error = %v, want ErrValidityRequired", err)
	}

	empty, err := series.NewNullable([]int{}, []bool{})
	if err != nil {
		t.Fatal(err)
	}
	if !empty.Nullable() {
		t.Fatal("empty Series created with validity is not nullable")
	}
}

func TestDropNullRemovesNullsAndKeepsDenseSeries(t *testing.T) {
	withNulls, err := series.NewNullable(
		[]int{2, 0, 4},
		[]bool{true, false, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	dropped := withNulls.DropNull()
	if dropped.Nullable() {
		t.Fatal("DropNull result is nullable")
	}
	if got, want := dropped.Values(), []int{2, 4}; !slices.Equal(got, want) {
		t.Fatalf("values = %v, want %v", got, want)
	}

	noNulls, err := series.NewNullable(
		[]int{2, 4},
		[]bool{true, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	dropped = noNulls.DropNull()
	if dropped.Nullable() {
		t.Fatal("DropNull of all-present nullable Series is nullable")
	}
	if got, want := dropped.Values(), []int{2, 4}; !slices.Equal(got, want) {
		t.Fatalf("all-present values = %v, want %v", got, want)
	}

	dense := series.New([]int{1, 2})
	dropped = dense.DropNull()
	if dropped.Nullable() {
		t.Fatal("DropNull of non-nullable Series is nullable")
	}
	if got, want := dropped.Values(), []int{1, 2}; !slices.Equal(got, want) {
		t.Fatalf("dense values = %v, want %v", got, want)
	}
}

func TestPresentRowsReturnsPresentIndexes(t *testing.T) {
	numbers, err := series.NewNullable(
		[]int{2, 99, 4, 88},
		[]bool{true, false, true, false},
	)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := numbers.PresentRows(), []int{0, 2}; !slices.Equal(got, want) {
		t.Fatalf("nullable rows = %v, want %v", got, want)
	}

	dense := series.New([]int{2, 4, 6})
	if got, want := dense.PresentRows(), []int{0, 1, 2}; !slices.Equal(got, want) {
		t.Fatalf("dense rows = %v, want %v", got, want)
	}

	allNull, err := series.NewNullable([]int{99}, []bool{false})
	if err != nil {
		t.Fatal(err)
	}
	if got := allNull.PresentRows(); len(got) != 0 {
		t.Fatalf("all-null rows = %v, want empty", got)
	}
}

func TestFillNullAlwaysReturnsNonNullable(t *testing.T) {
	withoutNulls, err := series.NewNullable(
		[]int{2, 4},
		[]bool{true, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	filled := withoutNulls.FillNull(99)
	if filled.Nullable() {
		t.Fatal("FillNull result is nullable")
	}
	if got, want := filled.Values(), []int{2, 4}; !slices.Equal(got, want) {
		t.Fatalf("values = %v, want %v", got, want)
	}

	withNulls, err := series.NewNullable(
		[]int{2, 0, 4},
		[]bool{true, false, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	filled = withNulls.FillNull(3)
	if filled.Nullable() {
		t.Fatal("FillNull result is nullable")
	}
	if got, want := filled.Values(), []int{2, 3, 4}; !slices.Equal(got, want) {
		t.Fatalf("values = %v, want %v", got, want)
	}
}

func TestMatchingRowsSkipsNulls(t *testing.T) {
	numbers, err := series.NewNullable(
		[]int{1, 2, 3, 4},
		[]bool{true, false, true, true},
	)
	if err != nil {
		t.Fatal(err)
	}

	calls := 0
	rows := numbers.MatchingRows(func(number int) bool {
		calls++
		return number%2 == 0
	})
	if got, want := rows, []int{3}; !slices.Equal(got, want) {
		t.Fatalf("rows = %v, want %v", got, want)
	}
	if calls != 3 {
		t.Fatalf("predicate calls = %d, want 3", calls)
	}
}

func TestGroupRowsPreservesOrderAndSeparatesNullFromZero(t *testing.T) {
	numbers, err := series.NewNullable(
		[]int{2, 0, 2, 99, 0, 88},
		[]bool{true, true, true, false, true, false},
	)
	if err != nil {
		t.Fatal(err)
	}
	allNull, err := series.NewNullable(
		[]int{99, 88},
		[]bool{false, false},
	)
	if err != nil {
		t.Fatal(err)
	}

	for name, test := range map[string]struct {
		values series.Series[int]
		want   [][]int
	}{
		"mixed": {
			values: numbers,
			want:   [][]int{{0, 2}, {1, 4}, {3, 5}},
		},
		"empty": {
			values: series.New([]int{}),
			want:   [][]int{},
		},
		"all null": {
			values: allNull,
			want:   [][]int{{0, 1}},
		},
	} {
		t.Run(name, func(t *testing.T) {
			got := series.GroupRows(test.values)
			if !slices.EqualFunc(got, test.want, func(left, right []int) bool {
				return slices.Equal(left, right)
			}) {
				t.Fatalf("groups = %v, want %v", got, test.want)
			}
		})
	}
}

func TestGroupRowsUsesGoEqualityForNaN(t *testing.T) {
	numbers := series.New([]float64{math.NaN(), math.NaN(), 1, 1})
	got := series.GroupRows(numbers)
	want := [][]int{{0}, {1}, {2, 3}}
	if !slices.EqualFunc(got, want, func(left, right []int) bool {
		return slices.Equal(left, right)
	}) {
		t.Fatalf("groups = %v, want %v", got, want)
	}
}

func TestUniqueRowsAndUniquePreserveFirstSeenOrderAndNullableSchema(t *testing.T) {
	numbers, err := series.NewNullable(
		[]int{2, 0, 2, 99, 0, 88, 3},
		[]bool{true, true, true, false, true, false, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	if got, want := series.UniqueRows(numbers), []int{0, 1, 3, 6}; !slices.Equal(got, want) {
		t.Fatalf("unique rows = %v, want %v", got, want)
	}

	unique := series.Unique(numbers)
	if got, want := unique.Values(), []int{2, 0, 99, 3}; !slices.Equal(got, want) {
		t.Fatalf("values = %v, want %v", got, want)
	}
	if got, want := unique.Validity(), []bool{true, true, false, true}; !slices.Equal(got, want) {
		t.Fatalf("validity = %v, want %v", got, want)
	}
	if !unique.Nullable() {
		t.Fatal("unique Series lost nullable schema")
	}
	if numbers.Len() != 7 {
		t.Fatalf("source length = %d, want 7", numbers.Len())
	}

	dense := series.Unique(series.New([]int{2, 2, 1}))
	if got, want := dense.Values(), []int{2, 1}; !slices.Equal(got, want) {
		t.Fatalf("dense values = %v, want %v", got, want)
	}
	if dense.Nullable() {
		t.Fatal("dense unique Series became nullable")
	}

	allPresent, err := series.NewNullable([]int{1, 1}, []bool{true, true})
	if err != nil {
		t.Fatal(err)
	}
	if result := series.Unique(allPresent); !result.Nullable() {
		t.Fatal("all-present unique Series lost nullable schema")
	}

	allNull, err := series.NewNullable([]int{99, 88}, []bool{false, false})
	if err != nil {
		t.Fatal(err)
	}
	if got, want := series.UniqueRows(allNull), []int{0}; !slices.Equal(got, want) {
		t.Fatalf("all-null rows = %v, want %v", got, want)
	}
	if got := series.UniqueRows(series.New([]int{})); len(got) != 0 {
		t.Fatalf("empty rows = %v, want empty", got)
	}
}

func TestUniqueRowsUsesGoEqualityForNaN(t *testing.T) {
	numbers := series.New([]float64{math.NaN(), math.NaN(), 1, 1})
	if got, want := series.UniqueRows(numbers), []int{0, 1, 2}; !slices.Equal(got, want) {
		t.Fatalf("rows = %v, want %v", got, want)
	}
}

func TestJoinRowsPreservesStableCartesianOrder(t *testing.T) {
	left := series.New([]int{2, 1, 2, 3})
	right := series.New([]int{2, 2, 1, 4})

	leftRows, rightRows := series.JoinRows(left, right)
	if want := []int{0, 0, 1, 2, 2}; !slices.Equal(leftRows, want) {
		t.Fatalf("left rows = %v, want %v", leftRows, want)
	}
	if want := []int{0, 1, 2, 0, 1}; !slices.Equal(rightRows, want) {
		t.Fatalf("right rows = %v, want %v", rightRows, want)
	}
}

func TestJoinRowsSkipsNullsAndMatchesPresentZero(t *testing.T) {
	left, err := series.NewNullable(
		[]int{0, 0, 1},
		[]bool{true, false, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	right, err := series.NewNullable(
		[]int{0, 0, 1},
		[]bool{true, false, true},
	)
	if err != nil {
		t.Fatal(err)
	}

	leftRows, rightRows := series.JoinRows(left, right)
	if want := []int{0, 2}; !slices.Equal(leftRows, want) {
		t.Fatalf("left rows = %v, want %v", leftRows, want)
	}
	if want := []int{0, 2}; !slices.Equal(rightRows, want) {
		t.Fatalf("right rows = %v, want %v", rightRows, want)
	}
}

func TestJoinRowsUsesGoEqualityForNaN(t *testing.T) {
	left := series.New([]float64{math.NaN(), 1})
	right := series.New([]float64{math.NaN(), 1})

	leftRows, rightRows := series.JoinRows(left, right)
	if want := []int{1}; !slices.Equal(leftRows, want) {
		t.Fatalf("left rows = %v, want %v", leftRows, want)
	}
	if want := []int{1}; !slices.Equal(rightRows, want) {
		t.Fatalf("right rows = %v, want %v", rightRows, want)
	}
}

func TestJoinRowsHandlesEmptyInputs(t *testing.T) {
	empty := series.New([]int{})
	values := series.New([]int{1})
	for name, test := range map[string]struct {
		left  series.Series[int]
		right series.Series[int]
	}{
		"empty left":  {left: empty, right: values},
		"empty right": {left: values, right: empty},
	} {
		t.Run(name, func(t *testing.T) {
			leftRows, rightRows := series.JoinRows(test.left, test.right)
			if len(leftRows) != 0 || len(rightRows) != 0 {
				t.Fatalf("rows = %v, %v; want no matches", leftRows, rightRows)
			}
		})
	}
}

func TestLeftJoinRowsPreservesMatchesAndUnmatchedRows(t *testing.T) {
	left, err := series.NewNullable(
		[]int{2, 0, 1, 2, 3},
		[]bool{true, false, true, true, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	right := series.New([]int{2, 2, 1})

	leftRows, rightRows := series.LeftJoinRows(left, right)
	if want := []int{0, 0, 1, 2, 3, 3, 4}; !slices.Equal(leftRows, want) {
		t.Fatalf("left rows = %v, want %v", leftRows, want)
	}
	rightValid := rightRows.Validity()
	if want := []bool{true, true, false, true, true, true, false}; !slices.Equal(rightValid, want) {
		t.Fatalf("right validity = %v, want %v", rightValid, want)
	}

	wantRightRows := []int{0, 1, 2, 0, 1}
	gotRightRows := make([]int, 0, len(wantRightRows))
	for i, valid := range rightValid {
		if valid {
			row, _ := rightRows.At(i)
			gotRightRows = append(gotRightRows, row)
		}
	}
	if !slices.Equal(gotRightRows, wantRightRows) {
		t.Fatalf("matched right rows = %v, want %v", gotRightRows, wantRightRows)
	}
}

func TestLeftJoinRowsUsesGoEqualityForNaN(t *testing.T) {
	left := series.New([]float64{math.NaN(), 1})
	right := series.New([]float64{math.NaN(), 1})

	leftRows, rightRows := series.LeftJoinRows(left, right)
	if want := []int{0, 1}; !slices.Equal(leftRows, want) {
		t.Fatalf("left rows = %v, want %v", leftRows, want)
	}
	rightValid := rightRows.Validity()
	if want := []bool{false, true}; !slices.Equal(rightValid, want) {
		t.Fatalf("right validity = %v, want %v", rightValid, want)
	}
	if row, valid := rightRows.At(1); row != 1 || !valid {
		t.Fatalf("matched right row = (%d, %v), want (1, true)", row, valid)
	}
}

func TestLeftJoinRowsHandlesEmptyInputs(t *testing.T) {
	empty := series.New([]int{})
	values := series.New([]int{1, 2})

	leftRows, rightRows := series.LeftJoinRows(empty, values)
	if len(leftRows) != 0 || rightRows.Len() != 0 {
		t.Fatalf("empty left rows = %v, %v; want empty results", leftRows, rightRows.Values())
	}
	if !rightRows.Nullable() {
		t.Fatal("empty left join row indexes are non-nullable")
	}

	leftRows, rightRows = series.LeftJoinRows(values, empty)
	if want := []int{0, 1}; !slices.Equal(leftRows, want) {
		t.Fatalf("empty right left rows = %v, want %v", leftRows, want)
	}
	if rightRows.Len() != 2 {
		t.Fatalf("empty right row count = %d, want 2", rightRows.Len())
	}
	if got, want := rightRows.Validity(), []bool{false, false}; !slices.Equal(got, want) {
		t.Fatalf("empty right validity = %v, want %v", got, want)
	}
}

func TestTakeNullableCombinesRowAndSourceValidity(t *testing.T) {
	values, err := series.NewNullable(
		[]int{10, 99, 20},
		[]bool{true, false, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	rows, err := series.NewNullable(
		[]int{2, 1, 0, 99},
		[]bool{true, true, false, false},
	)
	if err != nil {
		t.Fatal(err)
	}

	selected := values.TakeNullable(rows)
	if got, want := selected.Validity(), []bool{true, false, false, false}; !slices.Equal(got, want) {
		t.Fatalf("validity = %v, want %v", got, want)
	}
	if got, valid := selected.At(0); got != 20 || !valid {
		t.Fatalf("row 0 = (%d, %v), want (20, true)", got, valid)
	}
}

func TestSlicePreservesRowsAndNullableSchema(t *testing.T) {
	dense := series.New([]int{1, 2, 3, 4})
	denseSlice := dense.Slice(1, 3)
	if got, want := denseSlice.Values(), []int{2, 3}; !slices.Equal(got, want) {
		t.Fatalf("dense values = %v, want %v", got, want)
	}
	if denseSlice.Nullable() {
		t.Fatal("dense slice is nullable")
	}

	nullable, err := series.NewNullable(
		[]int{1, 99, 3, 4},
		[]bool{true, false, true, true},
	)
	if err != nil {
		t.Fatal(err)
	}
	nullableSlice := nullable.Slice(1, 3)
	if got, want := nullableSlice.Validity(), []bool{false, true}; !slices.Equal(got, want) {
		t.Fatalf("nullable validity = %v, want %v", got, want)
	}
	if got, valid := nullableSlice.At(1); got != 3 || !valid {
		t.Fatalf("nullable row 1 = (%d, %v), want (3, true)", got, valid)
	}

	empty := nullable.Slice(2, 2)
	if empty.Len() != 0 || !empty.Nullable() {
		t.Fatalf("empty slice = (len %d, nullable %v), want (0, true)", empty.Len(), empty.Nullable())
	}
	if dense.Len() != 4 || nullable.Len() != 4 {
		t.Fatalf("source lengths = %d, %d; want 4, 4", dense.Len(), nullable.Len())
	}
}

func TestSlicePanicsForInvalidBounds(t *testing.T) {
	numbers := series.New([]int{1, 2, 3})
	tests := map[string]struct {
		start int
		end   int
	}{
		"negative start": {start: -1, end: 1},
		"reversed":       {start: 2, end: 1},
		"past end":       {start: 0, end: 4},
	}
	for name, test := range tests {
		t.Run(name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("Slice did not panic")
				}
			}()
			numbers.Slice(test.start, test.end)
		})
	}
}

func TestSliceUsesLengthInsteadOfBackingCapacity(t *testing.T) {
	left := series.New([]int{1, 1, 1})
	right := series.New([]int{1, 1, 1})
	_, rows := series.LeftJoinRows(left, right)
	if rows.Len() != 9 {
		t.Fatalf("joined row count = %d, want 9", rows.Len())
	}

	defer func() {
		if recover() == nil {
			t.Fatal("Slice beyond Len did not panic")
		}
	}()
	rows.Slice(0, rows.Len()+1)
}

func TestSortedRowsIsStableAndSkipsNulls(t *testing.T) {
	numbers, err := series.NewNullable(
		[]int{2, 99, 1, 2, 88},
		[]bool{true, false, true, true, false},
	)
	if err != nil {
		t.Fatal(err)
	}

	compare := func(left, right int) int {
		if left == 99 || left == 88 || right == 99 || right == 88 {
			t.Fatal("comparator called with a null value")
		}
		switch {
		case left < right:
			return -1
		case left > right:
			return 1
		default:
			return 0
		}
	}

	if got, want := numbers.SortedRows(compare, false), []int{2, 0, 3, 1, 4}; !slices.Equal(got, want) {
		t.Fatalf("nulls-last rows = %v, want %v", got, want)
	}
	if got, want := numbers.SortedRows(compare, true), []int{1, 4, 2, 0, 3}; !slices.Equal(got, want) {
		t.Fatalf("nulls-first rows = %v, want %v", got, want)
	}
}
