package series

import (
	"errors"
	"math"
	"math/rand/v2"
	"runtime"
	"slices"
	"testing"

	"github.com/joeychilson/dataframe/internal/bitmap"
	"github.com/joeychilson/dataframe/mask"
)

func TestNew_CopiesValuesIntoNonNullableSeries(t *testing.T) {
	input := []int{10, 20, 30}
	values := New(input)
	input[0] = 99

	if values.Len() != 3 {
		t.Fatalf("Len() = %d, want 3", values.Len())
	}
	if values.Nullable() {
		t.Fatal("Nullable() = true, want false")
	}
	if values.NullCount() != 0 {
		t.Fatalf("NullCount() = %d, want 0", values.NullCount())
	}
	if got := values.Values(); !slices.Equal(got, []int{10, 20, 30}) {
		t.Fatalf("Values() = %v, want [10 20 30]", got)
	}
	if got := values.Validity(); !slices.Equal(got, []bool{true, true, true}) {
		t.Fatalf("Validity() = %v, want [true true true]", got)
	}
}

func TestNewFunc_CallsValueFunctionInRowOrder(t *testing.T) {
	var calls []int
	values := NewFunc(4, func(i int) int {
		calls = append(calls, i)
		return i * i
	})
	if got := values.Values(); !slices.Equal(got, []int{0, 1, 4, 9}) {
		t.Fatalf("NewFunc values = %v", got)
	}
	if !slices.Equal(calls, []int{0, 1, 2, 3}) || values.Nullable() {
		t.Fatalf("NewFunc calls = %v, nullable %t", calls, values.Nullable())
	}
	defer func() {
		if recover() == nil {
			t.Fatal("NewFunc did not panic for a negative length")
		}
	}()
	NewFunc(-1, func(int) int { return 0 })
}

func TestNewNullable_CopiesValuesAndValidity(t *testing.T) {
	input := []string{"first", "hidden", "third"}
	validity := []bool{true, false, true}
	values, err := NewNullable(input, validity)
	if err != nil {
		t.Fatalf("NewNullable() error = %v", err)
	}

	input[0] = "changed"
	validity[0] = false

	physical := values.Values()
	if physical[0] != "first" || physical[2] != "third" {
		t.Fatalf("Values() at present rows = %v, want copied input", physical)
	}
	if got := values.Validity(); !slices.Equal(got, []bool{true, false, true}) {
		t.Fatalf("Validity() = %v, want copied validity", got)
	}
	if values.NullCount() != 1 {
		t.Fatalf("NullCount() = %d, want 1", values.NullCount())
	}
}

func TestNewNullableFunc_CallsValueFunctionInRowOrder(t *testing.T) {
	var calls []int
	values := NewNullableFunc(4, func(i int) (int, bool) {
		calls = append(calls, i)
		return i * i, i%2 == 0
	})
	if got := values.Optionals(); !slices.Equal(got, []Optional[int]{Some(0), None[int](), Some(4), None[int]()}) {
		t.Fatalf("NewNullableFunc values = %+v", got)
	}
	if !slices.Equal(calls, []int{0, 1, 2, 3}) || !values.Nullable() {
		t.Fatalf("NewNullableFunc calls = %v, nullable %t", calls, values.Nullable())
	}
	allPresent := NewNullableFunc(2, func(i int) (int, bool) { return i, true })
	if !allPresent.Nullable() || allPresent.NullCount() != 0 {
		t.Fatalf("all-present NewNullableFunc = {Nullable:%t NullCount:%d}", allPresent.Nullable(), allPresent.NullCount())
	}
	if empty := NewNullableFunc(0, func(int) (int, bool) { panic("called") }); !empty.Nullable() || empty.Len() != 0 {
		t.Fatalf("empty NewNullableFunc = {Len:%d Nullable:%t}", empty.Len(), empty.Nullable())
	}
	defer func() {
		if recover() == nil {
			t.Fatal("NewNullableFunc did not panic for a negative length")
		}
	}()
	NewNullableFunc(-1, func(int) (int, bool) { return 0, false })
}

func TestFunctionConstructors_PanicOnNilFunction(t *testing.T) {
	tests := []struct {
		name string
		call func()
	}{
		{name: "NewFunc rejects a nil function", call: func() { NewFunc[int](0, nil) }},
		{name: "NewNullableFunc rejects a nil function", call: func() { NewNullableFunc[int](0, nil) }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("call did not panic")
				}
			}()
			test.call()
		})
	}
}

func TestNewNullablePreservesNullableSchema(t *testing.T) {
	tests := []struct {
		name   string
		values []int
		valid  []bool
	}{
		{name: "all-present validity remains nullable", values: []int{1, 2}, valid: []bool{true, true}},
		{name: "empty validity remains nullable", values: nil, valid: nil},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			values, err := NewNullable(test.values, test.valid)
			if err != nil {
				t.Fatalf("NewNullable() error = %v", err)
			}
			if !values.Nullable() {
				t.Fatal("Nullable() = false, want true")
			}
		})
	}
}

func TestNewNullable_RejectsLengthMismatch(t *testing.T) {
	values, err := NewNullable([]int{1, 2}, []bool{true})
	if !errors.Is(err, ErrLengthMismatch) {
		t.Fatalf("NewNullable() error = %v, want ErrLengthMismatch", err)
	}
	if values.Len() != 0 {
		t.Fatalf("NewNullable() Series length = %d, want 0", values.Len())
	}
}

func TestFromOptionals_CopiesValuesAndPreservesNulls(t *testing.T) {
	input := []Optional[int]{Some(10), {Value: 99}, Some(30)}
	values := FromOptionals(input)
	input[0] = None[int]()

	if got := values.Optionals(); !slices.Equal(got, []Optional[int]{Some(10), None[int](), Some(30)}) {
		t.Fatalf("Optionals() = %+v, want present, null, present", got)
	}
	if !values.Nullable() {
		t.Fatal("Nullable() = false, want true")
	}
}

func TestFromOptionalsPreservesNullableSchema(t *testing.T) {
	tests := []struct {
		name   string
		values []Optional[int]
	}{
		{name: "all-present optionals remain nullable", values: []Optional[int]{Some(1), Some(2)}},
		{name: "empty optionals remain nullable", values: nil},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if values := FromOptionals(test.values); !values.Nullable() {
				t.Fatal("Nullable() = false, want true")
			}
		})
	}
}

func TestRepeat_ConstructsRequestedCopies(t *testing.T) {
	values := Repeat("go", 3)
	if got := values.Values(); !slices.Equal(got, []string{"go", "go", "go"}) {
		t.Fatalf("Repeat(\"go\", 3).Values() = %v", got)
	}
	if values.NullCount() != 0 {
		t.Fatalf("NullCount() = %d, want 0", values.NullCount())
	}
	if values.Nullable() {
		t.Fatal("Nullable() = true, want false")
	}
}

func TestRepeatPanicsForNegativeCount(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("Repeat(value, -1) did not panic")
		}
	}()
	Repeat(1, -1)
}

func TestNullable_ZeroSeriesIsNonNullable(t *testing.T) {
	var values Series[int]
	if values.Nullable() {
		t.Fatal("zero Series Nullable() = true, want false")
	}
}

func TestSeriesAccess_DistinguishesPresentAndNullRows(t *testing.T) {
	values, err := NewNullable(
		[]string{"first", "physical null value", "third"},
		[]bool{true, false, true},
	)
	if err != nil {
		t.Fatalf("NewNullable() error = %v", err)
	}

	if value, ok := values.At(0); value != "first" || !ok {
		t.Fatalf("At(0) = (%q, %t), want (\"first\", true)", value, ok)
	}
	if value, ok := values.At(1); value != "" || ok {
		t.Fatalf("At(1) = (%q, %t), want (\"\", false)", value, ok)
	}
	if !values.IsValid(0) || values.IsValid(1) {
		t.Fatalf("IsValid() = [%t %t], want [true false]", values.IsValid(0), values.IsValid(1))
	}
	if value := values.ValueOr(0, "fallback"); value != "first" {
		t.Fatalf("ValueOr(0) = %q, want \"first\"", value)
	}
	if value := values.ValueOr(1, "fallback"); value != "fallback" {
		t.Fatalf("ValueOr(1) = %q, want \"fallback\"", value)
	}
}

func TestSeriesAccessPanicsOutOfRange(t *testing.T) {
	values := New([]int{1})
	tests := []struct {
		name string
		call func()
	}{
		{name: "IsValid rejects a negative index", call: func() { values.IsValid(-1) }},
		{name: "IsValid rejects an index equal to length", call: func() { values.IsValid(values.Len()) }},
		{name: "At rejects a negative index", call: func() { values.At(-1) }},
		{name: "At rejects an index equal to length", call: func() { values.At(values.Len()) }},
		{name: "ValueOr rejects a negative index", call: func() { values.ValueOr(-1, 0) }},
		{name: "ValueOr rejects an index equal to length", call: func() { values.ValueOr(values.Len(), 0) }},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("call did not panic")
				}
			}()
			test.call()
		})
	}
}

func TestFirstAndLastPresent_SkipNullRows(t *testing.T) {
	tests := []struct {
		name      string
		values    Series[int]
		wantFirst int
		wantLast  int
		wantOK    bool
	}{
		{name: "the zero value has no present rows", values: Series[int]{}},
		{name: "a non-null series returns its endpoints", values: New([]int{10, 20, 30}), wantFirst: 10, wantLast: 30, wantOK: true},
		{name: "a nullable series skips null endpoints", values: FromOptionals([]Optional[int]{None[int](), Some(20), None[int](), Some(40)}), wantFirst: 20, wantLast: 40, wantOK: true},
		{name: "an all-null series has no present rows", values: FromOptionals([]Optional[int]{None[int](), None[int]()})},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			first, firstOK := test.values.FirstPresent()
			if first != test.wantFirst || firstOK != test.wantOK {
				t.Errorf("FirstPresent() = (%d, %t), want (%d, %t)", first, firstOK, test.wantFirst, test.wantOK)
			}
			last, lastOK := test.values.LastPresent()
			if last != test.wantLast || lastOK != test.wantOK {
				t.Errorf("LastPresent() = (%d, %t), want (%d, %t)", last, lastOK, test.wantLast, test.wantOK)
			}
		})
	}
}

func TestSeriesSnapshotsAreIndependent(t *testing.T) {
	values := FromOptionals([]Optional[int]{Some(10), None[int](), Some(30)})

	physical := values.Values()
	validity := values.Validity()
	optionals := values.Optionals()
	physical[0] = 100
	validity[0] = false
	optionals[0] = None[int]()

	if value, ok := values.At(0); value != 10 || !ok {
		t.Fatalf("At(0) after mutating snapshots = (%d, %t), want (10, true)", value, ok)
	}
}

func TestSeriesImmutabilityIsShallow(t *testing.T) {
	referenced := []int{10}
	input := [][]int{referenced}
	values := New(input)

	input[0] = []int{20}
	value, _ := values.At(0)
	if value[0] != 10 {
		t.Fatalf("replacing input element changed Series value to %v", value)
	}

	referenced[0] = 30
	value, _ = values.At(0)
	if value[0] != 30 {
		t.Fatalf("referenced mutation did not reach Series value: %v", value)
	}

	snapshot := values.Values()
	snapshot[0][0] = 40
	value, _ = values.At(0)
	if value[0] != 40 {
		t.Fatalf("referenced snapshot mutation did not reach Series value: %v", value)
	}

	snapshot[0] = []int{50}
	value, _ = values.At(0)
	if value[0] != 40 {
		t.Fatalf("replacing snapshot element changed Series value to %v", value)
	}
}

func TestAll_IteratesEveryOptionalRow(t *testing.T) {
	values := FromOptionals([]Optional[string]{Some("first"), None[string](), Some("third")})

	var indexes []int
	var cells []Optional[string]
	for index, cell := range values.All() {
		indexes = append(indexes, index)
		cells = append(cells, cell)
	}

	if !slices.Equal(indexes, []int{0, 1, 2}) {
		t.Fatalf("All() indexes = %v, want [0 1 2]", indexes)
	}
	if !slices.Equal(cells, []Optional[string]{Some("first"), None[string](), Some("third")}) {
		t.Fatalf("All() cells = %+v, want present, null, present", cells)
	}

	yields := 0
	values.All()(func(int, Optional[string]) bool {
		yields++
		return false
	})
	if yields != 1 {
		t.Fatalf("All() yielded %d rows after false, want 1", yields)
	}
}

func TestPresent_IteratesOnlyPresentRows(t *testing.T) {
	values := FromOptionals([]Optional[int]{None[int](), Some(20), None[int](), Some(40)})

	var indexes []int
	var present []int
	for index, value := range values.Present() {
		indexes = append(indexes, index)
		present = append(present, value)
	}

	if !slices.Equal(indexes, []int{1, 3}) {
		t.Fatalf("Present() indexes = %v, want [1 3]", indexes)
	}
	if !slices.Equal(present, []int{20, 40}) {
		t.Fatalf("Present() values = %v, want [20 40]", present)
	}

	yields := 0
	values.Present()(func(int, int) bool {
		yields++
		return false
	})
	if yields != 1 {
		t.Fatalf("Present() yielded %d rows after false, want 1", yields)
	}
}

func TestEqualFunc_ComparesValidityAndPresentValues(t *testing.T) {
	hiddenLeft, err := NewNullable([]int{10, 20}, []bool{true, false})
	if err != nil {
		t.Fatal(err)
	}
	hiddenRight, err := NewNullable([]int{10, 99}, []bool{true, false})
	if err != nil {
		t.Fatal(err)
	}
	allPresent, err := NewNullable([]int{10, 20}, []bool{true, true})
	if err != nil {
		t.Fatal(err)
	}
	tests := []struct {
		name  string
		left  Series[int]
		right Series[int]
		want  bool
	}{
		{name: "equal values and validity compare equal", left: New([]int{10, 20}), right: New([]int{10, 20}), want: true},
		{name: "different lengths compare unequal", left: New([]int{10}), right: New([]int{10, 20})},
		{name: "different present values compare unequal", left: New([]int{10, 20}), right: New([]int{10, 30})},
		{name: "different validity compares unequal", left: hiddenLeft, right: allPresent},
		{name: "hidden values do not affect equality", left: hiddenLeft, right: hiddenRight, want: true},
		{name: "nullable schema does not affect equality", left: New([]int{10, 20}), right: allPresent, want: true},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := test.left.EqualFunc(test.right, func(left, right int) bool { return left == right }); got != test.want {
				t.Fatalf("EqualFunc() = %t, want %t", got, test.want)
			}
		})
	}

	left := New([][]int{{1, 2}, {3}})
	right := New([][]int{{1, 2}, {3}})
	if !left.EqualFunc(right, slices.Equal) {
		t.Fatal("EqualFunc() could not compare slice values")
	}
}

func TestEqualFuncPanicsForNilFunction(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("EqualFunc(nil) did not panic")
		}
	}()
	Series[int]{}.EqualFunc(Series[int]{}, nil)
}

func TestEqual_UsesGoEquality(t *testing.T) {
	if !Equal(New([]string{"go"}), New([]string{"go"})) {
		t.Fatal("Equal() = false for equal Series")
	}
	if Equal(New([]float64{math.NaN()}), New([]float64{math.NaN()})) {
		t.Fatal("Equal() matched NaN values")
	}
}

func TestConcat_AppendsValuesAndWidensNullability(t *testing.T) {
	left := New([]int{10, 20})
	middle := FromOptionals([]Optional[int]{None[int](), Some(40)})
	right := New([]int{50})
	result := left.Concat(middle, right)
	want := []Optional[int]{Some(10), Some(20), None[int](), Some(40), Some(50)}
	if got := result.Optionals(); !slices.Equal(got, want) {
		t.Fatalf("Concat() = %+v, want %+v", got, want)
	}
	if !result.Nullable() {
		t.Fatal("Concat() did not widen nullable schema")
	}

	dense := left.Concat(right)
	if dense.Nullable() || !slices.Equal(dense.Values(), []int{10, 20, 50}) {
		t.Fatalf("dense Concat() = {Values:%v Nullable:%t}", dense.Values(), dense.Nullable())
	}

	widened := left.Concat(FromOptionals[int](nil))
	if !widened.Nullable() || !slices.Equal(widened.Values(), []int{10, 20}) {
		t.Fatalf("Concat(nullable empty) = {Values:%v Nullable:%t}", widened.Values(), widened.Nullable())
	}

	unchanged := left.Concat()
	if &unchanged.values[0] != &left.values[0] {
		t.Fatal("Concat() copied its receiver with no inputs")
	}
}

func TestMap_TransformsPresentValues(t *testing.T) {
	source := FromOptionals([]Optional[int]{Some(10), None[int](), Some(30)})
	calls := 0
	mapped := source.Map(func(value int) string {
		calls++
		return string(rune('a' + value/10 - 1))
	})
	want := []Optional[string]{Some("a"), None[string](), Some("c")}
	if got := mapped.Optionals(); !slices.Equal(got, want) {
		t.Fatalf("Map() = %+v, want %+v", got, want)
	}
	if calls != 2 || !mapped.Nullable() {
		t.Fatalf("Map() = {Calls:%d Nullable:%t}", calls, mapped.Nullable())
	}

	dense := New([]int{1, 2}).Map(func(value int) int { return value * 2 })
	if dense.Nullable() || !slices.Equal(dense.Values(), []int{2, 4}) {
		t.Fatalf("dense Map() = {Values:%v Nullable:%t}", dense.Values(), dense.Nullable())
	}
}

func TestMapCells_TransformsEveryCell(t *testing.T) {
	source := FromOptionals([]Optional[int]{Some(10), None[int](), Some(30)})
	calls := 0
	mapped := source.MapCells(func(cell Optional[int]) Optional[string] {
		calls++
		if !cell.Valid {
			return Some("missing")
		}
		if cell.Value == 30 {
			return None[string]()
		}
		return Some("present")
	})
	want := []Optional[string]{Some("present"), Some("missing"), None[string]()}
	if got := mapped.Optionals(); !slices.Equal(got, want) {
		t.Fatalf("MapCells() = %+v, want %+v", got, want)
	}
	if calls != source.Len() || !mapped.Nullable() {
		t.Fatalf("MapCells() = {Calls:%d Nullable:%t}", calls, mapped.Nullable())
	}
}

func TestMapOptional_ControlsOutputValidity(t *testing.T) {
	source := FromOptionals([]Optional[int]{Some(10), None[int](), Some(30)})
	calls := 0
	mapped := source.MapOptional(func(value int) (int, bool) {
		calls++
		return value * 2, value != 30
	})
	want := []Optional[int]{Some(20), None[int](), None[int]()}
	if got := mapped.Optionals(); !slices.Equal(got, want) {
		t.Fatalf("MapOptional() = %+v, want %+v", got, want)
	}
	if calls != 2 || !mapped.Nullable() {
		t.Fatalf("MapOptional() = {Calls:%d Nullable:%t}", calls, mapped.Nullable())
	}
}

func TestTryMap_WrapsTheFailingRow(t *testing.T) {
	stop := errors.New("stop")
	source := FromOptionals([]Optional[int]{Some(10), None[int](), Some(30)})
	_, err := source.TryMap(func(value int) (int, error) {
		if value == 30 {
			return 0, stop
		}
		return value * 2, nil
	})
	if !errors.Is(err, stop) || err.Error() != "series: try map: row 2: stop" {
		t.Fatalf("TryMap() error = %v", err)
	}

	mapped, err := source.TryMap(func(value int) (int, error) { return value * 2, nil })
	if err != nil {
		t.Fatal(err)
	}
	if got := mapped.Optionals(); !slices.Equal(got, []Optional[int]{Some(20), None[int](), Some(60)}) {
		t.Fatalf("TryMap() = %+v", got)
	}
}

func TestTryMapCells_WrapsTheFailingRow(t *testing.T) {
	stop := errors.New("stop")
	source := FromOptionals([]Optional[int]{Some(10), None[int]()})
	_, err := source.TryMapCells(func(cell Optional[int]) (Optional[int], error) {
		if !cell.Valid {
			return None[int](), stop
		}
		return Some(cell.Value), nil
	})
	if !errors.Is(err, stop) || err.Error() != "series: try map cells: row 1: stop" {
		t.Fatalf("TryMapCells() error = %v", err)
	}
}

func TestMap2_CombinesPresentRows(t *testing.T) {
	left := FromOptionals([]Optional[int]{Some(10), None[int](), Some(30)})
	right := FromOptionals([]Optional[int]{Some(1), Some(2), None[int]()})
	calls := 0
	mapped := left.Map2(right, func(leftValue, rightValue int) int {
		calls++
		return leftValue + rightValue
	})
	if got := mapped.Optionals(); !slices.Equal(got, []Optional[int]{Some(11), None[int](), None[int]()}) {
		t.Fatalf("Map2() = %+v", got)
	}
	if calls != 1 || !mapped.Nullable() {
		t.Fatalf("Map2() = {Calls:%d Nullable:%t}", calls, mapped.Nullable())
	}

	dense := New([]int{1, 2}).Map2(New([]int{3, 4}), func(leftValue, rightValue int) int { return leftValue + rightValue })
	if dense.Nullable() || !slices.Equal(dense.Values(), []int{4, 6}) {
		t.Fatalf("dense Map2() = {Values:%v Nullable:%t}", dense.Values(), dense.Nullable())
	}
	leftNullable := left.Map2(New([]int{1, 2, 3}), func(leftValue, rightValue int) int { return leftValue + rightValue })
	if got := leftNullable.Optionals(); !slices.Equal(got, []Optional[int]{Some(11), None[int](), Some(33)}) {
		t.Fatalf("left-nullable Map2() = %+v", got)
	}
	rightNullable := New([]int{1, 2, 3}).Map2(right, func(leftValue, rightValue int) int { return leftValue + rightValue })
	if got := rightNullable.Optionals(); !slices.Equal(got, []Optional[int]{Some(2), Some(4), None[int]()}) {
		t.Fatalf("right-nullable Map2() = %+v", got)
	}
}

func TestMap2Cells_CombinesEveryCell(t *testing.T) {
	left := FromOptionals([]Optional[int]{Some(10), None[int]()})
	right := FromOptionals([]Optional[int]{None[int](), Some(2)})
	mapped := left.Map2Cells(right, func(leftCell, rightCell Optional[int]) Optional[int] {
		return Some(leftCell.Or(0) + rightCell.Or(0))
	})
	if got := mapped.Optionals(); !slices.Equal(got, []Optional[int]{Some(10), Some(2)}) {
		t.Fatalf("Map2Cells() = %+v", got)
	}
}

func TestTryMap2_WrapsTheFailingRow(t *testing.T) {
	stop := errors.New("stop")
	left := New([]int{1, 2})
	right := New([]int{10, 20})
	_, err := left.TryMap2(right, func(leftValue, rightValue int) (int, error) {
		if leftValue == 2 {
			return 0, stop
		}
		return leftValue + rightValue, nil
	})
	if !errors.Is(err, stop) || err.Error() != "series: try map 2: row 1: stop" {
		t.Fatalf("TryMap2() error = %v", err)
	}

	_, err = left.TryMap2Cells(right, func(leftCell, rightCell Optional[int]) (Optional[int], error) {
		if leftCell.Value == 2 {
			return None[int](), stop
		}
		return Some(leftCell.Value + rightCell.Value), nil
	})
	if !errors.Is(err, stop) || err.Error() != "series: try map 2 cells: row 1: stop" {
		t.Fatalf("TryMap2Cells() error = %v", err)
	}
}

func TestMap2LengthMismatchPanics(t *testing.T) {
	left := New([]int{1})
	right := New([]int{1, 2})
	tests := []struct {
		name string
		call func()
	}{
		{name: "Map2 rejects mismatched lengths", call: func() { left.Map2(right, func(a, b int) int { return a + b }) }},
		{name: "Map2Cells rejects mismatched lengths", call: func() { left.Map2Cells(right, func(a, b Optional[int]) Optional[int] { return a }) }},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("call did not panic")
				}
			}()
			test.call()
		})
	}
}

func TestTryMap2LengthMismatchReturnsError(t *testing.T) {
	left := New([]int{1})
	right := New([]int{1, 2})
	_, err := left.TryMap2(right, func(a, b int) (int, error) { return a + b, nil })
	if !errors.Is(err, ErrLengthMismatch) {
		t.Fatalf("TryMap2() length error = %v, want ErrLengthMismatch", err)
	}
	_, err = left.TryMap2Cells(right, func(a, b Optional[int]) (Optional[int], error) { return a, nil })
	if !errors.Is(err, ErrLengthMismatch) {
		t.Fatalf("TryMap2Cells() length error = %v, want ErrLengthMismatch", err)
	}
}

func TestTransformFunctions_PanicOnNilFunction(t *testing.T) {
	empty := Series[int]{}
	allNull := FromOptionals([]Optional[int]{None[int]()})
	tests := []struct {
		name string
		call func()
	}{
		{name: "Map rejects a nil function", call: func() { allNull.Map[int](nil) }},
		{name: "MapCells rejects a nil function", call: func() { empty.MapCells[int](nil) }},
		{name: "MapOptional rejects a nil function", call: func() { allNull.MapOptional[int](nil) }},
		{name: "TryMap rejects a nil function", call: func() { _, _ = allNull.TryMap[int](nil) }},
		{name: "TryMapCells rejects a nil function", call: func() { _, _ = empty.TryMapCells[int](nil) }},
		{name: "Map2 rejects a nil function", call: func() { allNull.Map2[int, int](allNull, nil) }},
		{name: "Map2Cells rejects a nil function", call: func() { empty.Map2Cells[int, int](empty, nil) }},
		{name: "TryMap2 rejects a nil function", call: func() { _, _ = allNull.TryMap2[int, int](allNull, nil) }},
		{name: "TryMap2Cells rejects a nil function", call: func() { _, _ = empty.TryMap2Cells[int, int](empty, nil) }},
		{name: "Scan rejects a nil function", call: func() { allNull.Scan[int](0, nil) }},
		{name: "Reduce rejects a nil function", call: func() { allNull.Reduce[int](0, nil) }},
		{name: "SortedFunc rejects a nil comparator", call: func() { allNull.SortedFunc(nil) }},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("call did not panic")
				}
			}()
			test.call()
		})
	}
}

func TestScanAndReduce_FoldPresentValues(t *testing.T) {
	values := FromOptionals([]Optional[int]{Some(1), None[int](), Some(3)})
	scanned := values.Scan(0, func(sum, value int) int { return sum + value })
	if got := scanned.Optionals(); !slices.Equal(got, []Optional[int]{Some(1), None[int](), Some(4)}) {
		t.Fatalf("Scan() = %+v", got)
	}
	if !scanned.Nullable() {
		t.Fatal("Scan() removed nullable schema")
	}
	if got := values.Reduce(0, func(sum, value int) int { return sum + value }); got != 4 {
		t.Fatalf("Reduce() = %d, want 4", got)
	}
}

func TestSortedFunc_OrdersPresentValuesAndNulls(t *testing.T) {
	dense := New([]int{1, 2, 3})
	if sorted := dense.SortedFunc(func(left, right int) int { return left - right }); &sorted.values[0] != &dense.values[0] {
		t.Fatal("SortedFunc(already sorted) copied a non-null Series")
	}
	nullable := FromOptionals([]Optional[int]{Some(1), Some(2), None[int]()})
	if sorted := nullable.SortedFunc(func(left, right int) int { return left - right }); &sorted.values[0] != &nullable.values[0] {
		t.Fatal("SortedFunc(already sorted) copied a nullable Series")
	}

	type item struct {
		key int
		id  string
	}
	values, err := NewNullable(
		[]item{{2, "a"}, {0, "null1"}, {1, "b"}, {2, "c"}, {0, "null2"}},
		[]bool{true, false, true, true, false},
	)
	if err != nil {
		t.Fatal(err)
	}
	sorted := values.SortedFunc(func(left, right item) int {
		if left.id == "null1" || left.id == "null2" || right.id == "null1" || right.id == "null2" {
			t.Fatal("comparator received a null physical value")
		}
		return left.key - right.key
	})
	gotValues := sorted.Values()
	gotIDs := make([]string, len(gotValues))
	for i, value := range gotValues {
		gotIDs[i] = value.id
	}
	if !slices.Equal(gotIDs, []string{"b", "a", "c", "null1", "null2"}) {
		t.Fatalf("SortedFunc() order = %v", gotIDs)
	}
	if got := sorted.Validity(); !slices.Equal(got, []bool{true, true, true, false, false}) {
		t.Fatalf("SortedFunc() validity = %v", got)
	}
	if got := values.Values(); got[0].id != "a" {
		t.Fatal("SortedFunc() changed source")
	}
}

func TestFilter_KeepsSelectedRowsInOrder(t *testing.T) {
	source := FromOptionals([]Optional[int]{Some(10), None[int](), Some(30), None[int]()})
	identity := source.Filter(mask.All(source.Len()))
	if &identity.values[0] != &source.values[0] {
		t.Fatal("Filter(all) copied an immutable Series")
	}
	tests := []struct {
		name      string
		selection mask.Mask
		want      []Optional[int]
	}{
		{
			name:      "keeps selected null rows",
			selection: mask.New([]bool{true, true, false, false}),
			want:      []Optional[int]{Some(10), None[int]()},
		},
		{
			name:      "keeps selected present rows",
			selection: mask.New([]bool{true, false, true, false}),
			want:      []Optional[int]{Some(10), Some(30)},
		},
		{
			name:      "returns empty when no rows are selected",
			selection: mask.None(source.Len()),
			want:      nil,
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			filtered := source.Filter(test.selection)
			if got := filtered.Optionals(); !slices.Equal(got, test.want) {
				t.Fatalf("Filter() = %+v, want %+v", got, test.want)
			}
			if !filtered.Nullable() {
				t.Fatal("Filter() removed nullable schema")
			}
		})
	}

	dense := New([]int{10, 20, 30}).Filter(mask.New([]bool{true, false, true}))
	if dense.Nullable() {
		t.Fatal("Filter() made a non-null Series nullable")
	}
	if got := dense.Values(); !slices.Equal(got, []int{10, 30}) {
		t.Fatalf("non-null Filter() = %v, want [10 30]", got)
	}
	if got := source.Optionals(); !slices.Equal(got, []Optional[int]{Some(10), None[int](), Some(30), None[int]()}) {
		t.Fatalf("Filter() changed source to %+v", got)
	}
}

func TestFilterPanicsOnLengthMismatch(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("Filter did not panic")
		}
	}()
	New([]int{1, 2}).Filter(mask.All(1))
}

func TestTake_KeepsRequestedRowsInOrder(t *testing.T) {
	source := FromOptionals([]Optional[int]{Some(10), None[int](), Some(30)})
	identity := source.Take([]int{0, 1, 2})
	if &identity.values[0] != &source.values[0] {
		t.Fatal("Take(identity) copied an immutable Series")
	}
	taken := source.Take([]int{2, 1, 2, 0})
	want := []Optional[int]{Some(30), None[int](), Some(30), Some(10)}
	if got := taken.Optionals(); !slices.Equal(got, want) {
		t.Fatalf("Take() = %+v, want %+v", got, want)
	}
	if !taken.Nullable() {
		t.Fatal("Take() removed nullable schema")
	}

	empty := source.Take(nil)
	if empty.Len() != 0 || !empty.Nullable() {
		t.Fatalf("empty Take() = {Len:%d Nullable:%t}", empty.Len(), empty.Nullable())
	}

	dense := New([]int{10, 20, 30}).Take([]int{2, 0})
	if dense.Nullable() {
		t.Fatal("Take() made a non-null Series nullable")
	}
	if got := dense.Values(); !slices.Equal(got, []int{30, 10}) {
		t.Fatalf("non-null Take() = %v, want [30 10]", got)
	}
}

func TestTakePanicsOnInvalidIndex(t *testing.T) {
	values := New([]int{10, 20})
	tests := []struct {
		name string
		rows []int
	}{
		{name: "rejects a negative index", rows: []int{-1}},
		{name: "rejects an index equal to length", rows: []int{values.Len()}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("Take did not panic")
				}
			}()
			values.Take(test.rows)
		})
	}
}

func TestTakeNullable_PropagatesIndexAndSourceNulls(t *testing.T) {
	source := FromOptionals([]Optional[int]{Some(10), None[int](), Some(30)})
	rows := FromOptionals([]Optional[int]{Some(2), None[int](), Some(1), Some(0)})
	taken := source.TakeNullable(rows)
	want := []Optional[int]{Some(30), None[int](), None[int](), Some(10)}
	if got := taken.Optionals(); !slices.Equal(got, want) {
		t.Fatalf("TakeNullable() = %+v, want %+v", got, want)
	}
	if !taken.Nullable() {
		t.Fatal("TakeNullable() returned a non-null Series")
	}

	dense := New([]int{10, 20}).TakeNullable(New([]int{1, 0}))
	if got := dense.Optionals(); !slices.Equal(got, []Optional[int]{Some(20), Some(10)}) || !dense.Nullable() {
		t.Fatalf("dense TakeNullable() = %+v, nullable %t", got, dense.Nullable())
	}
}

func TestTakeNullablePanicsOnInvalidPresentIndex(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("TakeNullable did not panic")
		}
	}()
	New([]int{10}).TakeNullable(FromOptionals([]Optional[int]{Some(-1), None[int]()}))
}

func TestHeadAndTail_ClampCountsAndShareStorage(t *testing.T) {
	values := New([]int{10, 20, 30, 40})
	tests := []struct {
		name string
		got  Series[int]
		want []int
	}{
		{name: "Head returns empty for a zero count", got: values.Head(0), want: nil},
		{name: "Head returns the requested prefix", got: values.Head(2), want: []int{10, 20}},
		{name: "Head clamps a count past length", got: values.Head(10), want: []int{10, 20, 30, 40}},
		{name: "Tail returns empty for a zero count", got: values.Tail(0), want: nil},
		{name: "Tail returns the requested suffix", got: values.Tail(2), want: []int{30, 40}},
		{name: "Tail clamps a count past length", got: values.Tail(10), want: []int{10, 20, 30, 40}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := test.got.Values(); !slices.Equal(got, test.want) {
				t.Fatalf("values = %v, want %v", got, test.want)
			}
		})
	}

	nullable := FromOptionals([]Optional[int]{Some(10), None[int]()})
	if !nullable.Head(0).Nullable() || !nullable.Tail(0).Nullable() {
		t.Fatal("Head or Tail removed nullable schema from an empty result")
	}
}

func TestHeadAndTailPanicOnNegativeCount(t *testing.T) {
	values := New([]int{10, 20})
	tests := []struct {
		name string
		call func()
	}{
		{name: "Head rejects a negative count", call: func() { values.Head(-1) }},
		{name: "Tail rejects a negative count", call: func() { values.Tail(-1) }},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("call did not panic")
				}
			}()
			test.call()
		})
	}
}

func TestSlice_ReturnsSharedBoundedView(t *testing.T) {
	source := FromOptionals([]Optional[int]{Some(10), None[int](), Some(30), Some(40)})
	sliced := source.Slice(1, 3)
	if got := sliced.Optionals(); !slices.Equal(got, []Optional[int]{None[int](), Some(30)}) {
		t.Fatalf("Slice(1, 3) = %+v, want null, 30", got)
	}
	if !sliced.Nullable() {
		t.Fatal("Slice() removed nullable schema")
	}
	if got := slices.Collect(sliced.IsNull().Rows()); !slices.Equal(got, []int{0}) {
		t.Fatalf("Slice().IsNull() rows = %v, want [0]", got)
	}
	if got := slices.Collect(sliced.IsNotNull().Rows()); !slices.Equal(got, []int{1}) {
		t.Fatalf("Slice().IsNotNull() rows = %v, want [1]", got)
	}
	if &sliced.values[0] != &source.values[1] {
		t.Fatal("Slice() did not share value storage")
	}
	source.validity.Set(2, false)
	if sliced.IsValid(1) {
		t.Fatal("Slice() did not share validity storage")
	}
	if cap(sliced.values) != sliced.Len() {
		t.Fatal("Slice() values can extend into adjacent source rows")
	}

	empty := source.Slice(2, 2)
	if empty.Len() != 0 || !empty.Nullable() {
		t.Fatalf("empty Slice() = {Len:%d Nullable:%t}", empty.Len(), empty.Nullable())
	}
}

func TestSlicePanicsOnInvalidBounds(t *testing.T) {
	values := Series[int]{values: make([]int, 3, 4)}
	tests := []struct {
		name       string
		start, end int
	}{
		{name: "rejects a negative start", start: -1, end: 1},
		{name: "rejects reversed bounds", start: 2, end: 1},
		{name: "rejects an end past length", start: 0, end: 4},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("Slice did not panic")
				}
			}()
			values.Slice(test.start, test.end)
		})
	}
}

func TestNullMasks_SelectNullAndPresentRows(t *testing.T) {
	tests := []struct {
		name   string
		values Series[int]
		nulls  []bool
	}{
		{name: "empty non-null values produce empty masks", values: Series[int]{}, nulls: nil},
		{name: "empty nullable values produce empty masks", values: FromOptionals[int](nil), nulls: nil},
		{name: "non-null values mark every row present", values: New([]int{10, 20, 30}), nulls: []bool{false, false, false}},
		{name: "all-present nullable values mark every row present", values: FromOptionals([]Optional[int]{Some(10), Some(20)}), nulls: []bool{false, false}},
		{name: "mixed values distinguish null and present rows", values: FromOptionals([]Optional[int]{Some(10), None[int](), Some(30)}), nulls: []bool{false, true, false}},
		{name: "all-null values mark every row null", values: FromOptionals([]Optional[int]{None[int](), None[int]()}), nulls: []bool{true, true}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			nulls := test.values.IsNull()
			present := test.values.IsNotNull()
			if nulls.Len() != test.values.Len() || present.Len() != test.values.Len() {
				t.Fatalf("mask lengths = (%d, %d), want %d", nulls.Len(), present.Len(), test.values.Len())
			}
			for i, wantNull := range test.nulls {
				if got := nulls.At(i); got != wantNull {
					t.Errorf("IsNull().At(%d) = %t, want %t", i, got, wantNull)
				}
				wantPresent := !wantNull
				if got := present.At(i); got != wantPresent {
					t.Errorf("IsNotNull().At(%d) = %t, want %t", i, got, wantPresent)
				}
			}
		})
	}
}

func TestFillNull_ReplacesNullRows(t *testing.T) {
	tests := []struct {
		name   string
		values Series[int]
		want   []int
		share  bool
	}{
		{name: "empty non-null values remain empty", values: Series[int]{}, want: nil, share: true},
		{name: "empty nullable values remain empty", values: FromOptionals[int](nil), want: nil, share: true},
		{name: "non-null values reuse existing storage", values: New([]int{10, 20}), want: []int{10, 20}, share: true},
		{name: "all-present nullable values reuse existing storage", values: FromOptionals([]Optional[int]{Some(10), Some(20)}), want: []int{10, 20}, share: true},
		{name: "mixed values replace only null rows", values: FromOptionals([]Optional[int]{Some(10), None[int](), Some(30)}), want: []int{10, 99, 30}},
		{name: "all-null values replace every row", values: FromOptionals([]Optional[int]{None[int](), None[int]()}), want: []int{99, 99}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			before := test.values.Optionals()
			filled := test.values.FillNull(99)
			if filled.Nullable() {
				t.Fatal("FillNull() returned a nullable Series")
			}
			if got := filled.Values(); !slices.Equal(got, test.want) {
				t.Fatalf("FillNull(99) = %v, want %v", got, test.want)
			}
			if test.share && test.values.Len() != 0 && &filled.values[0] != &test.values.values[0] {
				t.Fatal("FillNull() copied values when no rows changed")
			}
			if got := test.values.Optionals(); !slices.Equal(got, before) {
				t.Fatalf("FillNull() changed source from %+v to %+v", before, got)
			}
		})
	}
}

func TestDropNull_KeepsPresentRows(t *testing.T) {
	tests := []struct {
		name   string
		values Series[int]
		want   []int
		share  bool
	}{
		{name: "empty non-null values remain empty", values: Series[int]{}, want: nil, share: true},
		{name: "empty nullable values remain empty", values: FromOptionals[int](nil), want: nil, share: true},
		{name: "non-null values reuse existing storage", values: New([]int{10, 20}), want: []int{10, 20}, share: true},
		{name: "all-present nullable values reuse existing storage", values: FromOptionals([]Optional[int]{Some(10), Some(20)}), want: []int{10, 20}, share: true},
		{name: "mixed values remove null rows", values: FromOptionals([]Optional[int]{Some(10), None[int](), Some(30)}), want: []int{10, 30}},
		{name: "all-null values produce an empty series", values: FromOptionals([]Optional[int]{None[int](), None[int]()}), want: nil},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			before := test.values.Optionals()
			dropped := test.values.DropNull()
			if dropped.Nullable() {
				t.Fatal("DropNull() returned a nullable Series")
			}
			if got := dropped.Values(); !slices.Equal(got, test.want) {
				t.Fatalf("DropNull() = %v, want %v", got, test.want)
			}
			if test.share && test.values.Len() != 0 && &dropped.values[0] != &test.values.values[0] {
				t.Fatal("DropNull() copied values when no rows changed")
			}
			if got := test.values.Optionals(); !slices.Equal(got, before) {
				t.Fatalf("DropNull() changed source from %+v to %+v", before, got)
			}
		})
	}
}

func FuzzRepresentationOperationsAgainstOptionals(f *testing.F) {
	boundary := make([]byte, 130)
	for i := range boundary {
		boundary[i] = byte(i)
	}
	f.Add(boundary, []byte{0, 1, 64, 129}, boundary, boundary, uint16(1), uint16(129))
	f.Add([]byte{}, []byte{}, []byte{}, []byte{}, uint16(0), uint16(0))

	f.Fuzz(func(t *testing.T, data, rowData, filterData, otherData []byte, rawStart, rawEnd uint16) {
		decode := func(encoded []byte) []Optional[int] {
			encoded = encoded[:min(len(encoded), 130)]
			values := make([]Optional[int], len(encoded))
			for i, value := range encoded {
				if value&1 != 0 {
					values[i] = Some(int(int8(value)))
				}
			}
			return values
		}
		assert := func(name string, got Series[int], want []Optional[int]) {
			t.Helper()
			if values := got.Optionals(); !slices.Equal(values, want) {
				t.Fatalf("%s = %+v, want %+v", name, values, want)
			}
		}

		model := decode(data)
		values := FromOptionals(model)
		start := int(rawStart) % (len(model) + 1)
		end := int(rawEnd) % (len(model) + 1)
		if start > end {
			start, end = end, start
		}
		assert("Slice", values.Slice(start, end), model[start:end])

		rowData = rowData[:min(len(rowData), 130)]
		rows := make([]int, 0, len(rowData))
		var wantTake []Optional[int]
		if len(model) > 0 {
			for _, value := range rowData {
				row := int(value) % len(model)
				rows = append(rows, row)
				wantTake = append(wantTake, model[row])
			}
		}
		assert("Take", values.Take(rows), wantTake)

		nullableRows := make([]Optional[int], len(rowData))
		wantNullable := make([]Optional[int], len(rowData))
		for i, value := range rowData {
			if len(model) > 0 && value&1 != 0 {
				row := int(value>>1) % len(model)
				nullableRows[i] = Some(row)
				wantNullable[i] = model[row]
			}
		}
		assert("TakeNullable", values.TakeNullable(FromOptionals(nullableRows)), wantNullable)

		selected := make([]bool, len(model))
		var wantFiltered []Optional[int]
		for i := range selected {
			selected[i] = i < len(filterData) && filterData[i]&1 != 0
			if selected[i] {
				wantFiltered = append(wantFiltered, model[i])
			}
		}
		assert("Filter", values.Filter(mask.New(selected)), wantFiltered)

		other := decode(otherData)
		wantConcat := append(slices.Clone(model), other...)
		assert("Concat", values.Concat(FromOptionals(other)), wantConcat)

		nulls := values.IsNull()
		present := values.IsNotNull()
		for i, value := range model {
			if nulls.At(i) != !value.Valid || present.At(i) != value.Valid {
				t.Fatalf("null masks at %d = (%t, %t), want (%t, %t)", i, nulls.At(i), present.At(i), !value.Valid, value.Valid)
			}
		}
		assert("source", values, model)
	})
}

func BenchmarkNew(b *testing.B) {
	input := make([]int, 1024)
	b.ReportAllocs()

	var result Series[int]
	for b.Loop() {
		result = New(input)
	}
	runtime.KeepAlive(result)
}

func BenchmarkNewFunc(b *testing.B) {
	const length = 1 << 16
	b.ReportAllocs()
	var result Series[int]
	for b.Loop() {
		result = NewFunc(length, func(i int) int { return i })
	}
	runtime.KeepAlive(result)
}

func BenchmarkNewNullable(b *testing.B) {
	input := make([]int, 1024)
	validity := slices.Repeat([]bool{true}, len(input))
	b.ReportAllocs()

	var result Series[int]
	for b.Loop() {
		var err error
		result, err = NewNullable(input, validity)
		if err != nil {
			b.Fatal(err)
		}
	}
	runtime.KeepAlive(result)
}

func BenchmarkNewNullableFunc(b *testing.B) {
	const length = 1 << 16
	b.ReportAllocs()
	var result Series[int]
	for b.Loop() {
		result = NewNullableFunc(length, func(i int) (int, bool) { return i, i%4 != 0 })
	}
	runtime.KeepAlive(result)
}

func BenchmarkPresent(b *testing.B) {
	input := make([]Optional[int], 1024)
	for i := range input {
		if i%4 != 0 {
			input[i] = Some(i)
		}
	}
	allPresent, err := NewNullable(
		slices.Repeat([]int{1}, len(input)),
		slices.Repeat([]bool{true}, len(input)),
	)
	if err != nil {
		b.Fatal(err)
	}
	benchmarks := []struct {
		name   string
		values Series[int]
	}{
		{name: "non-null", values: Repeat(1, len(input))},
		{name: "nullable/all-present", values: allPresent},
		{name: "nullable/25%-null", values: FromOptionals(input)},
	}

	for _, benchmark := range benchmarks {
		b.Run(benchmark.name, func(b *testing.B) {
			b.Run("Optimized", func(b *testing.B) {
				b.ReportAllocs()
				total := 0
				for b.Loop() {
					for _, value := range benchmark.values.Present() {
						total += value
					}
				}
				runtime.KeepAlive(total)
			})
			b.Run("Reference", func(b *testing.B) {
				b.ReportAllocs()
				total := 0
				for b.Loop() {
					for row := range benchmark.values.Len() {
						value, present := benchmark.values.At(row)
						if present {
							total += value
						}
					}
				}
				runtime.KeepAlive(total)
			})
		})
	}
}

func BenchmarkFilter(b *testing.B) {
	const length = 1 << 16
	values := make([]int, length)
	validity := make([]bool, length)
	sparseSelection := make([]bool, length)
	for i := range values {
		values[i] = i
		validity[i] = i%4 != 0
		sparseSelection[i] = i%16 == 0
	}
	nullable, err := NewNullable(values, validity)
	if err != nil {
		b.Fatal(err)
	}
	implementations := []struct {
		name   string
		filter func(Series[int], mask.Mask) Series[int]
	}{
		{name: "Optimized", filter: func(input Series[int], selection mask.Mask) Series[int] {
			return input.Filter(selection)
		}},
		{name: "Reference", filter: func(input Series[int], selection mask.Mask) Series[int] {
			selected := selection.Count()
			result := Series[int]{values: make([]int, selected)}
			if input.validity.Initialized() {
				result.validity = bitmap.New(selected)
			}
			index := 0
			for row := range selection.Rows() {
				result.values[index] = input.values[row]
				if result.validity.Initialized() && input.validity.At(row) {
					result.validity.Set(index, true)
				}
				index++
			}
			return result
		}},
	}
	benchmarks := []struct {
		name      string
		values    Series[int]
		selection mask.Mask
	}{
		{name: "non-null/dense", values: New(values), selection: mask.All(length)},
		{name: "non-null/sparse", values: New(values), selection: mask.New(sparseSelection)},
		{name: "nullable/dense", values: nullable, selection: mask.All(length)},
	}

	for _, benchmark := range benchmarks {
		b.Run(benchmark.name, func(b *testing.B) {
			for _, implementation := range implementations {
				b.Run(implementation.name, func(b *testing.B) {
					b.ReportAllocs()
					var result Series[int]
					for b.Loop() {
						result = implementation.filter(benchmark.values, benchmark.selection)
					}
					runtime.KeepAlive(result)
				})
			}
		})
	}
}

func BenchmarkTake(b *testing.B) {
	const length = 1 << 16
	rows := make([]int, length)
	for i := range rows {
		rows[i] = i
	}
	shuffled := slices.Clone(rows)
	rand.New(rand.NewPCG(1, 2)).Shuffle(len(shuffled), func(i, j int) {
		shuffled[i], shuffled[j] = shuffled[j], shuffled[i]
	})
	values := Repeat(1, length)
	implementations := []struct {
		name string
		take func([]int) Series[int]
	}{
		{name: "Optimized", take: values.Take},
		{name: "Reference", take: func(indexes []int) Series[int] {
			result := Series[int]{values: make([]int, len(indexes))}
			if values.validity.Initialized() {
				result.validity = bitmap.New(len(indexes))
			}
			for i, row := range indexes {
				result.values[i] = values.values[row]
				if result.validity.Initialized() && values.validity.At(row) {
					result.validity.Set(i, true)
				}
			}
			return result
		}},
	}
	benchmarks := []struct {
		name string
		rows []int
	}{
		{name: "ordered", rows: rows},
		{name: "shuffled", rows: shuffled},
	}

	for _, benchmark := range benchmarks {
		b.Run(benchmark.name, func(b *testing.B) {
			for _, implementation := range implementations {
				b.Run(implementation.name, func(b *testing.B) {
					b.ReportAllocs()
					var result Series[int]
					for b.Loop() {
						result = implementation.take(benchmark.rows)
					}
					runtime.KeepAlive(result)
				})
			}
		})
	}
}

func BenchmarkTakeNullable(b *testing.B) {
	const length = 1 << 16
	values := Repeat(1, length)
	rows := make([]Optional[int], length)
	for i := range rows {
		if i%4 != 0 {
			rows[i] = Some(i)
		}
	}
	indexes := FromOptionals(rows)
	b.ReportAllocs()
	var result Series[int]
	for b.Loop() {
		result = values.TakeNullable(indexes)
	}
	runtime.KeepAlive(result)
}

func BenchmarkSliceViews(b *testing.B) {
	values := Repeat(1, 1<<16)
	benchmarks := []struct {
		name string
		view func() Series[int]
	}{
		{name: "Slice", view: func() Series[int] { return values.Slice(1, values.Len()-1) }},
		{name: "Head", view: func() Series[int] { return values.Head(values.Len() / 2) }},
		{name: "Tail", view: func() Series[int] { return values.Tail(values.Len() / 2) }},
	}

	for _, benchmark := range benchmarks {
		b.Run(benchmark.name, func(b *testing.B) {
			b.ReportAllocs()
			var result Series[int]
			for b.Loop() {
				result = benchmark.view()
			}
			runtime.KeepAlive(result)
		})
	}
}

func BenchmarkNullMasks(b *testing.B) {
	const length = 1 << 16
	values := make([]int, length)
	allPresent := make([]bool, length)
	partiallyNull := make([]bool, length)
	for i := range values {
		allPresent[i] = true
		partiallyNull[i] = i%4 != 0
	}
	allPresentValues, err := NewNullable(values, allPresent)
	if err != nil {
		b.Fatal(err)
	}
	partiallyNullValues, err := NewNullable(values, partiallyNull)
	if err != nil {
		b.Fatal(err)
	}
	allNullValues, err := NewNullable(values, make([]bool, length))
	if err != nil {
		b.Fatal(err)
	}
	inputs := []struct {
		name   string
		values Series[int]
	}{
		{name: "non-null", values: New(values)},
		{name: "nullable/all-present", values: allPresentValues},
		{name: "nullable/25%-null", values: partiallyNullValues},
		{name: "nullable/all-null", values: allNullValues},
	}
	operations := []struct {
		name string
		mask func(Series[int]) mask.Mask
	}{
		{name: "IsNull", mask: func(inputValues Series[int]) mask.Mask { return inputValues.IsNull() }},
		{name: "IsNotNull", mask: func(inputValues Series[int]) mask.Mask { return inputValues.IsNotNull() }},
	}

	for _, operation := range operations {
		b.Run(operation.name, func(b *testing.B) {
			for _, input := range inputs {
				b.Run(input.name, func(b *testing.B) {
					b.ReportAllocs()
					var result mask.Mask
					for b.Loop() {
						result = operation.mask(input.values)
					}
					runtime.KeepAlive(result)
				})
			}
		})
	}
}

func BenchmarkNullTransforms(b *testing.B) {
	const length = 1 << 16
	values := make([]int, length)
	allPresent := make([]bool, length)
	partiallyNull := make([]bool, length)
	for i := range values {
		values[i] = i
		allPresent[i] = true
		partiallyNull[i] = i%4 != 0
	}
	allPresentValues, err := NewNullable(values, allPresent)
	if err != nil {
		b.Fatal(err)
	}
	partiallyNullValues, err := NewNullable(values, partiallyNull)
	if err != nil {
		b.Fatal(err)
	}
	allNullValues, err := NewNullable(values, make([]bool, length))
	if err != nil {
		b.Fatal(err)
	}
	inputs := []struct {
		name   string
		values Series[int]
	}{
		{name: "non-null", values: New(values)},
		{name: "nullable/all-present", values: allPresentValues},
		{name: "nullable/25%-null", values: partiallyNullValues},
		{name: "nullable/all-null", values: allNullValues},
	}
	operations := []struct {
		name      string
		transform func(Series[int]) Series[int]
	}{
		{name: "FillNull", transform: func(inputValues Series[int]) Series[int] { return inputValues.FillNull(0) }},
		{name: "DropNull", transform: func(inputValues Series[int]) Series[int] { return inputValues.DropNull() }},
	}

	for _, operation := range operations {
		b.Run(operation.name, func(b *testing.B) {
			for _, input := range inputs {
				b.Run(input.name, func(b *testing.B) {
					b.ReportAllocs()
					var result Series[int]
					for b.Loop() {
						result = operation.transform(input.values)
					}
					runtime.KeepAlive(result)
				})
			}
		})
	}
}

func BenchmarkConcat(b *testing.B) {
	dense := make([]Series[int], 8)
	for i := range dense {
		dense[i] = Repeat(i, 1<<10)
	}
	values := make([]int, (1<<10)+1)
	validity := make([]bool, len(values))
	for i := range validity {
		validity[i] = i%3 != 0
	}
	nullable, err := NewNullable(values, validity)
	if err != nil {
		b.Fatal(err)
	}
	sliced := make([]Series[int], 8)
	for i := range sliced {
		sliced[i] = nullable.Slice(1, nullable.Len())
	}

	for _, benchmark := range []struct {
		name  string
		parts []Series[int]
	}{
		{name: "Dense", parts: dense},
		{name: "NullableSliced", parts: sliced},
	} {
		b.Run(benchmark.name, func(b *testing.B) {
			first := benchmark.parts[0]
			rest := benchmark.parts[1:]
			b.ReportAllocs()
			var result Series[int]
			for b.Loop() {
				result = first.Concat(rest...)
			}
			runtime.KeepAlive(result)
		})
	}
}

func BenchmarkMap(b *testing.B) {
	const length = 1 << 16
	validity := make([]bool, length)
	for i := range validity {
		validity[i] = i%4 != 0
	}
	nullable, err := NewNullable(make([]int, length), validity)
	if err != nil {
		b.Fatal(err)
	}
	mapValue := func(value int) int { return value + 1 }
	referenceMap := func(input Series[int], fn func(int) int) Series[int] {
		result := Series[int]{values: make([]int, input.Len())}
		if input.validity.Initialized() {
			result.validity = input.validity.Clone()
		}
		for row, value := range input.values {
			if !input.validity.Initialized() || input.validity.At(row) {
				result.values[row] = fn(value)
			}
		}
		return result
	}
	for _, benchmark := range []struct {
		name   string
		values Series[int]
	}{
		{name: "NonNull", values: Repeat(1, length)},
		{name: "Nullable", values: nullable},
	} {
		b.Run(benchmark.name, func(b *testing.B) {
			b.Run("Optimized", func(b *testing.B) {
				b.ReportAllocs()
				var result Series[int]
				for b.Loop() {
					result = benchmark.values.Map(mapValue)
				}
				runtime.KeepAlive(result)
			})
			b.Run("Reference", func(b *testing.B) {
				b.ReportAllocs()
				var result Series[int]
				for b.Loop() {
					result = referenceMap(benchmark.values, mapValue)
				}
				runtime.KeepAlive(result)
			})
		})
	}
}

func BenchmarkCellTransforms(b *testing.B) {
	const length = 1 << 16
	physical := make([]int, length)
	validity := make([]bool, length)
	for i := range physical {
		physical[i] = i
		validity[i] = i%4 != 0
	}
	partiallyNull, err := NewNullable(physical, validity)
	if err != nil {
		b.Fatal(err)
	}
	inputs := []struct {
		name   string
		values Series[int]
	}{
		{name: "NonNull", values: New(physical)},
		{name: "PartiallyNull", values: partiallyNull},
	}
	for _, input := range inputs {
		b.Run("All/"+input.name, func(b *testing.B) {
			b.ReportAllocs()
			total := 0
			for b.Loop() {
				for _, cell := range input.values.All() {
					if cell.Valid {
						total += cell.Value
					}
				}
			}
			runtime.KeepAlive(total)
		})
		b.Run("MapCells/"+input.name, func(b *testing.B) {
			b.ReportAllocs()
			var result Series[int]
			for b.Loop() {
				result = input.values.MapCells(func(cell Optional[int]) Optional[int] {
					return cell
				})
			}
			runtime.KeepAlive(result)
		})
		b.Run("Map2Cells/"+input.name, func(b *testing.B) {
			b.ReportAllocs()
			var result Series[int]
			for b.Loop() {
				result = input.values.Map2Cells(input.values, func(left, right Optional[int]) Optional[int] {
					if !left.Valid || !right.Valid {
						return None[int]()
					}
					return Some(left.Value + right.Value)
				})
			}
			runtime.KeepAlive(result)
		})
	}
}

func BenchmarkSortedFunc(b *testing.B) {
	const length = 1 << 14
	sortedValues := make([]int, length)
	reverseValues := make([]int, length)
	mostlyPresent := make([]bool, length)
	nullHeavy := make([]bool, length)
	for i := range sortedValues {
		sortedValues[i] = i
		reverseValues[i] = length - i
		mostlyPresent[i] = i < length*3/4
		nullHeavy[i] = i < length/4
	}
	mostlyPresentValues, err := NewNullable(sortedValues, mostlyPresent)
	if err != nil {
		b.Fatal(err)
	}
	nullHeavyValues, err := NewNullable(sortedValues, nullHeavy)
	if err != nil {
		b.Fatal(err)
	}
	compare := func(leftValue, rightValue int) int { return leftValue - rightValue }
	implementations := []struct {
		name string
		sort func(Series[int]) Series[int]
	}{
		{name: "Optimized", sort: func(input Series[int]) Series[int] {
			return input.SortedFunc(compare)
		}},
		{name: "Reference", sort: func(input Series[int]) Series[int] {
			if !input.validity.Initialized() {
				values := slices.Clone(input.values)
				slices.SortStableFunc(values, compare)
				return Series[int]{values: values}
			}
			compareRows := func(leftRow, rightRow int) int {
				leftValid := input.validity.At(leftRow)
				rightValid := input.validity.At(rightRow)
				switch {
				case !leftValid && !rightValid:
					return 0
				case !leftValid:
					return 1
				case !rightValid:
					return -1
				default:
					return compare(input.values[leftRow], input.values[rightRow])
				}
			}
			rows := make([]int, input.Len())
			for i := range rows {
				rows[i] = i
			}
			slices.SortStableFunc(rows, compareRows)
			return input.Take(rows)
		}},
	}
	for _, benchmark := range []struct {
		name   string
		values Series[int]
	}{
		{name: "NonNull/Sorted", values: New(sortedValues)},
		{name: "NonNull/Reverse", values: New(reverseValues)},
		{name: "Nullable/MostlyPresent", values: mostlyPresentValues},
		{name: "Nullable/NullHeavy", values: nullHeavyValues},
	} {
		b.Run(benchmark.name, func(b *testing.B) {
			for _, implementation := range implementations {
				b.Run(implementation.name, func(b *testing.B) {
					b.ReportAllocs()
					var result Series[int]
					for b.Loop() {
						result = implementation.sort(benchmark.values)
					}
					runtime.KeepAlive(result)
				})
			}
		})
	}
}
