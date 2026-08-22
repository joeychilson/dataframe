package series

import (
	"hash/maphash"
	"math"
	"runtime"
	"slices"
	"testing"
)

type stringSliceHasher struct{}

func (stringSliceHasher) Hash(hash *maphash.Hash, values []string) {
	for _, value := range values {
		hash.WriteString(value)
		hash.WriteByte(0)
	}
}

func (stringSliceHasher) Equal(left, right []string) bool {
	return slices.Equal(left, right)
}

func TestRowComparisons(t *testing.T) {
	left := FromOptionals([]Optional[int]{Some(1), None[int](), Some(3)})
	right := New([]int{1, 2, 4})
	tests := []struct {
		name string
		mask func() []int
		want []int
	}{
		{name: "EqualRows", mask: func() []int { return slices.Collect(EqualRows(left, right).Rows()) }, want: []int{0}},
		{name: "NotEqualRows", mask: func() []int { return slices.Collect(NotEqualRows(left, right).Rows()) }, want: []int{2}},
		{name: "LessRows", mask: func() []int { return slices.Collect(LessRows(left, right).Rows()) }, want: []int{2}},
		{name: "LessEqualRows", mask: func() []int { return slices.Collect(LessEqualRows(left, right).Rows()) }, want: []int{0, 2}},
		{name: "GreaterRows", mask: func() []int { return slices.Collect(GreaterRows(left, right).Rows()) }, want: nil},
		{name: "GreaterEqualRows", mask: func() []int { return slices.Collect(GreaterEqualRows(left, right).Rows()) }, want: []int{0}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := test.mask(); !slices.Equal(got, test.want) {
				t.Fatalf("rows = %v, want %v", got, test.want)
			}
		})
	}
}

func TestRowComparisonsPanicOnLengthMismatch(t *testing.T) {
	short := New([]int{1})
	long := New([]int{1, 2})
	tests := []struct {
		name string
		call func()
	}{
		{name: "EqualRows", call: func() { EqualRows(short, long) }},
		{name: "NotEqualRows", call: func() { NotEqualRows(short, long) }},
		{name: "LessRows", call: func() { LessRows(short, long) }},
		{name: "LessEqualRows", call: func() { LessEqualRows(short, long) }},
		{name: "GreaterRows", call: func() { GreaterRows(short, long) }},
		{name: "GreaterEqualRows", call: func() { GreaterEqualRows(short, long) }},
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

func BenchmarkEqualRows(b *testing.B) {
	left := Repeat(1, 1<<16)
	right := Repeat(1, left.Len())
	b.ReportAllocs()
	var count int
	for b.Loop() {
		count = EqualRows(left, right).Count()
	}
	runtime.KeepAlive(count)
}

func TestValueComparisons(t *testing.T) {
	values := FromOptionals([]Optional[int]{Some(1), None[int](), Some(2), Some(3)})
	tests := []struct {
		name string
		rows []int
		want []int
	}{
		{name: "EqualValue", rows: slices.Collect(EqualValue(values, 2).Rows()), want: []int{2}},
		{name: "NotEqualValue", rows: slices.Collect(NotEqualValue(values, 2).Rows()), want: []int{0, 3}},
		{name: "LessValue", rows: slices.Collect(LessValue(values, 2).Rows()), want: []int{0}},
		{name: "LessEqualValue", rows: slices.Collect(LessEqualValue(values, 2).Rows()), want: []int{0, 2}},
		{name: "GreaterValue", rows: slices.Collect(GreaterValue(values, 2).Rows()), want: []int{3}},
		{name: "GreaterEqualValue", rows: slices.Collect(GreaterEqualValue(values, 2).Rows()), want: []int{2, 3}},
		{name: "Between", rows: slices.Collect(Between(values, 2, 3).Rows()), want: []int{2, 3}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if !slices.Equal(test.rows, test.want) {
				t.Fatalf("rows = %v, want %v", test.rows, test.want)
			}
		})
	}
}

func TestBetweenPanicsForReversedBounds(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("Between(2, 1) did not panic")
		}
	}()
	Between(New([]int{1}), 2, 1)
}

func TestIn(t *testing.T) {
	values := FromOptionals([]Optional[string]{Some("go"), None[string](), Some("rust"), Some("zig")})
	if rows := slices.Collect(In(values, "zig", "go", "go").Rows()); !slices.Equal(rows, []int{0, 3}) {
		t.Fatalf("In() rows = %v, want [0 3]", rows)
	}
	if In(values).Any() {
		t.Fatal("In() with no values selected a row")
	}
	nan := math.NaN()
	if In(New([]float64{nan}), nan).Any() {
		t.Fatal("In() matched NaN using ==")
	}
}

func TestInUsing(t *testing.T) {
	values := FromOptionals([]Optional[[]string]{
		Some([]string{"go", "rust"}),
		None[[]string](),
		Some([]string{"zig"}),
	})
	selection := InUsing(values, stringSliceHasher{}, []string{"zig"})
	if rows := slices.Collect(selection.Rows()); !slices.Equal(rows, []int{2}) {
		t.Fatalf("InUsing() rows = %v, want [2]", rows)
	}
}

func TestIsNaN(t *testing.T) {
	values := FromOptionals([]Optional[float64]{Some(1.0), Some(math.NaN()), None[float64]()})
	if rows := slices.Collect(IsNaN(values).Rows()); !slices.Equal(rows, []int{1}) {
		t.Fatalf("IsNaN() rows = %v, want [1]", rows)
	}
}

func TestArithmetic(t *testing.T) {
	left := FromOptionals([]Optional[int]{Some(12), None[int](), Some(8)})
	right := FromOptionals([]Optional[int]{Some(3), Some(2), Some(4)})
	tests := []struct {
		name string
		got  Series[int]
		want []Optional[int]
	}{
		{name: "Add", got: Add(left, right), want: []Optional[int]{Some(15), None[int](), Some(12)}},
		{name: "Sub", got: Sub(left, right), want: []Optional[int]{Some(9), None[int](), Some(4)}},
		{name: "Mul", got: Mul(left, right), want: []Optional[int]{Some(36), None[int](), Some(32)}},
		{name: "Div", got: Div(left, right), want: []Optional[int]{Some(4), None[int](), Some(2)}},
		{name: "Neg", got: Neg(left), want: []Optional[int]{Some(-12), None[int](), Some(-8)}},
		{name: "Abs", got: Abs(Neg(left)), want: []Optional[int]{Some(12), None[int](), Some(8)}},
		{name: "AddScalar", got: AddScalar(left, 2), want: []Optional[int]{Some(14), None[int](), Some(10)}},
		{name: "SubScalar", got: SubScalar(left, 2), want: []Optional[int]{Some(10), None[int](), Some(6)}},
		{name: "MulScalar", got: MulScalar(left, 2), want: []Optional[int]{Some(24), None[int](), Some(16)}},
		{name: "DivScalar", got: DivScalar(left, 2), want: []Optional[int]{Some(6), None[int](), Some(4)}},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if got := test.got.Optionals(); !slices.Equal(got, test.want) {
				t.Fatalf("result = %+v, want %+v", got, test.want)
			}
		})
	}

	negativeZero := math.Copysign(0, -1)
	absolute, _ := Abs(New([]float64{negativeZero})).At(0)
	if math.Signbit(absolute) {
		t.Fatal("Abs(-0) retained the negative sign")
	}
}

func TestIntegerDivisionByZero(t *testing.T) {
	defer func() {
		if recover() == nil {
			t.Fatal("Div by zero did not panic")
		}
	}()
	Div(New([]int{1}), New([]int{0}))
}

func BenchmarkAdd(b *testing.B) {
	left := Repeat(1, 1<<16)
	right := Repeat(2, left.Len())
	b.ReportAllocs()
	var result Series[int]
	for b.Loop() {
		result = Add(left, right)
	}
	runtime.KeepAlive(result)
}

func TestFloatTransforms(t *testing.T) {
	values := New([]float64{1, 4})
	if got := Sqrt(values).Values(); !slices.Equal(got, []float64{1, 2}) {
		t.Fatalf("Sqrt() = %v", got)
	}
	if got := Exp(New([]float64{0})).Values()[0]; got != 1 {
		t.Fatalf("Exp(0) = %v", got)
	}
	if got := Log(New([]float64{1})).Values()[0]; got != 0 {
		t.Fatalf("Log(1) = %v", got)
	}
	if got := Pow(New([]float64{2}), 3).Values()[0]; got != 8 {
		t.Fatalf("Pow(2, 3) = %v", got)
	}
}

func TestBasicAggregates(t *testing.T) {
	values := FromOptionals([]Optional[int]{Some(3), None[int](), Some(1), Some(2)})
	if got, ok := Sum(values); got != 6 || !ok {
		t.Fatalf("Sum() = (%d, %t), want (6, true)", got, ok)
	}
	if got, ok := Mean(values); got != 2 || !ok {
		t.Fatalf("Mean() = (%v, %t), want (2, true)", got, ok)
	}
	if got, ok := Min(values); got != 1 || !ok {
		t.Fatalf("Min() = (%d, %t), want (1, true)", got, ok)
	}
	if got, ok := Max(values); got != 3 || !ok {
		t.Fatalf("Max() = (%d, %t), want (3, true)", got, ok)
	}
	if got, ok := ArgMin(values); got != 2 || !ok {
		t.Fatalf("ArgMin() = (%d, %t), want (2, true)", got, ok)
	}
	if got, ok := ArgMax(values); got != 0 || !ok {
		t.Fatalf("ArgMax() = (%d, %t), want (0, true)", got, ok)
	}

	empty := FromOptionals[int](nil)
	if _, ok := Sum(empty); ok {
		t.Fatal("Sum(empty) reported a value")
	}
	if _, ok := Mean(empty); ok {
		t.Fatal("Mean(empty) reported a value")
	}
	if _, ok := Min(empty); ok {
		t.Fatal("Min(empty) reported a value")
	}
	if _, ok := Max(empty); ok {
		t.Fatal("Max(empty) reported a value")
	}
	if _, ok := ArgMin(empty); ok {
		t.Fatal("ArgMin(empty) reported a value")
	}
	if _, ok := ArgMax(empty); ok {
		t.Fatal("ArgMax(empty) reported a value")
	}
}

func TestNaNAggregates(t *testing.T) {
	values := New([]float64{2, math.NaN(), 1, math.NaN()})
	if got, _ := Min(values); !math.IsNaN(got) {
		t.Fatalf("Min() = %v, want NaN", got)
	}
	if got, _ := Max(values); !math.IsNaN(got) {
		t.Fatalf("Max() = %v, want NaN", got)
	}
	if got, _ := ArgMin(values); got != 1 {
		t.Fatalf("ArgMin() = %d, want 1", got)
	}
	if got, _ := ArgMax(values); got != 1 {
		t.Fatalf("ArgMax() = %d, want 1", got)
	}
}

func TestVariance(t *testing.T) {
	values := FromOptionals([]Optional[int]{Some(1), None[int](), Some(2), Some(3)})
	if got, ok := SampleVariance(values); got != 1 || !ok {
		t.Fatalf("SampleVariance() = (%v, %t), want (1, true)", got, ok)
	}
	if got, ok := SampleStdDev(values); got != 1 || !ok {
		t.Fatalf("SampleStdDev() = (%v, %t), want (1, true)", got, ok)
	}
	if _, ok := SampleVariance(New([]int{1})); ok {
		t.Fatal("SampleVariance(one value) reported a result")
	}
}

func TestVarianceExtremeValues(t *testing.T) {
	maximum := math.MaxFloat64
	for _, values := range [][]float64{
		{-maximum, maximum},
		{maximum, -maximum},
		{-maximum, 0, maximum},
	} {
		if got, ok := SampleVariance(New(values)); !ok || !math.IsInf(got, 1) {
			t.Errorf("SampleVariance(%v) = (%v, %t), want (+Inf, true)", values, got, ok)
		}
	}

	if got, ok := SampleVariance(New([]float64{maximum, maximum})); !ok || got != 0 {
		t.Errorf("SampleVariance(equal maximums) = (%v, %t), want (0, true)", got, ok)
	}

	values := make([]float64, 1000)
	values[0] = 1e155
	if got, ok := SampleVariance(New(values)); !ok || math.Abs(got-1e307)/1e307 > 1e-14 {
		t.Errorf("SampleVariance(large finite values) = (%v, %t), want approximately (1e307, true)", got, ok)
	}

	nullable := FromOptionals([]Optional[float64]{Some(-maximum), None[float64](), Some(maximum)})
	if got, ok := SampleVariance(nullable); !ok || !math.IsInf(got, 1) {
		t.Errorf("SampleVariance(nullable extremes) = (%v, %t), want (+Inf, true)", got, ok)
	}
	if got, ok := SampleStdDev(nullable); !ok || !math.IsInf(got, 1) {
		t.Errorf("SampleStdDev(nullable extremes) = (%v, %t), want (+Inf, true)", got, ok)
	}

	for _, nonFinite := range []float64{math.NaN(), math.Inf(1), math.Inf(-1)} {
		if got, ok := SampleVariance(New([]float64{0, nonFinite})); !ok || !math.IsNaN(got) {
			t.Errorf("SampleVariance([0, %v]) = (%v, %t), want (NaN, true)", nonFinite, got, ok)
		}
	}
}

func BenchmarkSampleVariance(b *testing.B) {
	values := make([]float64, 1<<14)
	for i := range values {
		values[i] = float64(i % 1000)
	}
	source := New(values)
	b.ReportAllocs()
	var result float64
	for b.Loop() {
		result, _ = SampleVariance(source)
	}
	runtime.KeepAlive(result)
}

func TestQuantileAndMedian(t *testing.T) {
	values := FromOptionals([]Optional[int]{Some(0), Some(10), None[int](), Some(20), Some(30)})
	tests := []struct {
		q    float64
		want float64
	}{
		{q: 0, want: 0},
		{q: 0.25, want: 7.5},
		{q: 0.5, want: 15},
		{q: 1, want: 30},
	}
	for _, test := range tests {
		if got, ok := Quantile(values, test.q); got != test.want || !ok {
			t.Errorf("Quantile(%v) = (%v, %t), want (%v, true)", test.q, got, ok, test.want)
		}
	}
	if got, ok := Median(values); got != 15 || !ok {
		t.Fatalf("Median() = (%v, %t), want (15, true)", got, ok)
	}
	if _, ok := Quantile(FromOptionals[int](nil), 0.5); ok {
		t.Fatal("Quantile(empty) reported a value")
	}
	if got, ok := Quantile(New([]float64{1, math.NaN()}), 0.5); !ok || !math.IsNaN(got) {
		t.Fatalf("Quantile(with NaN) = (%v, %t), want (NaN, true)", got, ok)
	}
	if got, ok := Quantile(New([]float64{math.Inf(1)}), 0.5); !ok || !math.IsInf(got, 1) {
		t.Fatalf("Quantile(+Inf) = (%v, %t), want (+Inf, true)", got, ok)
	}
	if got, ok := Median(New([]float64{-math.MaxFloat64, math.MaxFloat64})); !ok || got != 0 {
		t.Fatalf("Median(extreme finite values) = (%v, %t), want (0, true)", got, ok)
	}
}

func TestQuantilePanicsForInvalidQ(t *testing.T) {
	for _, q := range []float64{-0.1, 1.1, math.NaN()} {
		t.Run("invalid", func(t *testing.T) {
			defer func() {
				if recover() == nil {
					t.Fatal("Quantile did not panic")
				}
			}()
			Quantile(New([]int{1}), q)
		})
	}
}

func BenchmarkQuantile(b *testing.B) {
	values := make([]float64, 1<<14)
	for i := range values {
		values[i] = float64(len(values) - i)
	}
	source := New(values)
	b.ReportAllocs()
	var result float64
	for b.Loop() {
		result, _ = Quantile(source, 0.5)
	}
	runtime.KeepAlive(result)
}

func TestCumSum(t *testing.T) {
	values := FromOptionals([]Optional[int]{Some(1), None[int](), Some(3)})
	if got := CumSum(values).Optionals(); !slices.Equal(got, []Optional[int]{Some(1), None[int](), Some(4)}) {
		t.Fatalf("CumSum() = %+v", got)
	}
}

func TestSorted(t *testing.T) {
	values := FromOptionals([]Optional[int]{Some(2), None[int](), Some(1), Some(2)})
	if got := Sorted(values).Optionals(); !slices.Equal(got, []Optional[int]{Some(1), Some(2), Some(2), None[int]()}) {
		t.Fatalf("Sorted() = %+v", got)
	}
	if got := SortedDescending(values).Optionals(); !slices.Equal(got, []Optional[int]{Some(2), Some(2), Some(1), None[int]()}) {
		t.Fatalf("SortedDescending() = %+v", got)
	}
}
