package series

import (
	"cmp"
	"hash/maphash"
	"math"
	"slices"

	"github.com/joeychilson/dataframe/internal/hashmap"
	"github.com/joeychilson/dataframe/internal/reduce"
	"github.com/joeychilson/dataframe/mask"
)

// Integer permits built-in integer types, including defined types with those
// underlying types.
type Integer interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 | ~uintptr
}

// Float permits floating-point types.
type Float interface {
	~float32 | ~float64
}

// Real permits integer and floating-point types.
type Real interface {
	Integer | Float
}

// Number permits real and complex numeric types.
type Number interface {
	Real | ~complex64 | ~complex128
}

// SignedNumber permits signed integers and floating-point values. It is used
// where an unsigned or complex result would have surprising semantics, such as
// Abs.
type SignedNumber interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 | ~float32 | ~float64
}

// EqualRows compares a and b row by row using ==. Nulls never match. It panics
// on length mismatch. Use Equal to compare two complete Series values.
func EqualRows[T comparable](a, b Series[T]) mask.Mask {
	return matchRows(a, b, "series: EqualRows: length mismatch", func(a, b T) bool { return a == b })
}

// NotEqualRows compares a and b row by row using !=. Nulls never match. It
// panics on length mismatch.
func NotEqualRows[T comparable](a, b Series[T]) mask.Mask {
	return matchRows(a, b, "series: NotEqualRows: length mismatch", func(a, b T) bool { return a != b })
}

// LessRows compares a < b row by row. Nulls never match. It panics on length
// mismatch.
func LessRows[T cmp.Ordered](a, b Series[T]) mask.Mask {
	return matchRows(a, b, "series: LessRows: length mismatch", func(a, b T) bool { return a < b })
}

// LessEqualRows compares a <= b row by row. Nulls never match. It panics on
// length mismatch.
func LessEqualRows[T cmp.Ordered](a, b Series[T]) mask.Mask {
	return matchRows(a, b, "series: LessEqualRows: length mismatch", func(a, b T) bool { return a <= b })
}

// GreaterRows compares a > b row by row. Nulls never match. It panics on length
// mismatch.
func GreaterRows[T cmp.Ordered](a, b Series[T]) mask.Mask {
	return matchRows(a, b, "series: GreaterRows: length mismatch", func(a, b T) bool { return a > b })
}

// GreaterEqualRows compares a >= b row by row. Nulls never match. It panics on
// length mismatch.
func GreaterEqualRows[T cmp.Ordered](a, b Series[T]) mask.Mask {
	return matchRows(a, b, "series: GreaterEqualRows: length mismatch", func(a, b T) bool { return a >= b })
}

// EqualValue selects rows equal to value.
func EqualValue[T comparable](s Series[T], value T) mask.Mask {
	return matchValue(s, func(element T) bool { return element == value })
}

// NotEqualValue selects rows unequal to value.
func NotEqualValue[T comparable](s Series[T], value T) mask.Mask {
	return matchValue(s, func(element T) bool { return element != value })
}

// LessValue selects rows less than value.
func LessValue[T cmp.Ordered](s Series[T], value T) mask.Mask {
	return matchValue(s, func(element T) bool { return element < value })
}

// LessEqualValue selects rows less than or equal to value.
func LessEqualValue[T cmp.Ordered](s Series[T], value T) mask.Mask {
	return matchValue(s, func(element T) bool { return element <= value })
}

// GreaterValue selects rows greater than value.
func GreaterValue[T cmp.Ordered](s Series[T], value T) mask.Mask {
	return matchValue(s, func(element T) bool { return element > value })
}

// GreaterEqualValue selects rows greater than or equal to value.
func GreaterEqualValue[T cmp.Ordered](s Series[T], value T) mask.Mask {
	return matchValue(s, func(element T) bool { return element >= value })
}

// Between selects rows in the inclusive range [lo, hi]. It panics when hi is
// less than lo.
func Between[T cmp.Ordered](s Series[T], lo, hi T) mask.Mask {
	if hi < lo {
		panic("series: Between: upper bound is less than lower bound")
	}
	return mask.NewFunc(s.Len(), func(i int) bool {
		return (s.validity == nil || s.validity[i]) && lo <= s.values[i] && s.values[i] <= hi
	})
}

// In selects rows whose value occurs in values using ==.
func In[T comparable](s Series[T], values ...T) mask.Mask {
	if len(values) == 0 {
		return mask.None(s.Len())
	}
	set := make(map[T]struct{}, len(values))
	for _, value := range values {
		set[value] = struct{}{}
	}
	return mask.NewFunc(s.Len(), func(i int) bool {
		if s.validity != nil && !s.validity[i] {
			return false
		}
		_, ok := set[s.values[i]]
		return ok
	})
}

// InUsing selects rows whose value occurs in values using hasher's equivalence
// relation. It supports non-comparable values and custom equality and panics
// when hasher is nil.
func InUsing[T any](s Series[T], hasher maphash.Hasher[T], values ...T) mask.Mask {
	set := hashmap.New[T, struct{}](hasher)
	for _, value := range values {
		set.Set(value, struct{}{})
	}
	return mask.NewFunc(s.Len(), func(i int) bool {
		if s.validity != nil && !s.validity[i] {
			return false
		}
		_, ok := set.Get(s.values[i])
		return ok
	})
}

// IsNaN selects present NaN values.
func IsNaN[T Float](s Series[T]) mask.Mask {
	return mask.NewFunc(s.Len(), func(i int) bool {
		return (s.validity == nil || s.validity[i]) && s.values[i] != s.values[i]
	})
}

// Add computes a[i] + b[i]. Nulls propagate, integer overflow follows Go, and
// a length mismatch panics.
func Add[T Number](a, b Series[T]) Series[T] {
	return a.Map2(b, func(a, b T) T { return a + b })
}

// Sub computes a[i] - b[i]. It panics on length mismatch.
func Sub[T Number](a, b Series[T]) Series[T] {
	return a.Map2(b, func(a, b T) T { return a - b })
}

// Mul computes a[i] * b[i]. It panics on length mismatch.
func Mul[T Number](a, b Series[T]) Series[T] {
	return a.Map2(b, func(a, b T) T { return a * b })
}

// Div computes a[i] / b[i] using Go division semantics. Integer division by
// zero panics; floating-point and complex division follow Go's IEEE 754
// semantics. It panics on length mismatch.
func Div[T Number](a, b Series[T]) Series[T] {
	return a.Map2(b, func(a, b T) T { return a / b })
}

// Neg computes -s for signed values.
func Neg[T SignedNumber](s Series[T]) Series[T] {
	return s.Map(func(value T) T { return -value })
}

// Abs computes the absolute value of s. The minimum signed integer follows Go
// overflow semantics.
func Abs[T SignedNumber](s Series[T]) Series[T] {
	return s.Map(func(value T) T {
		if value == 0 {
			var zero T
			return zero
		}
		if value < 0 {
			return -value
		}
		return value
	})
}

// AddScalar computes s[i] + value.
func AddScalar[T Number](s Series[T], value T) Series[T] {
	return s.Map(func(element T) T { return element + value })
}

// SubScalar computes s[i] - value.
func SubScalar[T Number](s Series[T], value T) Series[T] {
	return s.Map(func(element T) T { return element - value })
}

// MulScalar computes s[i] * value.
func MulScalar[T Number](s Series[T], value T) Series[T] {
	return s.Map(func(element T) T { return element * value })
}

// DivScalar computes s[i] / value using Go division semantics.
func DivScalar[T Number](s Series[T], value T) Series[T] {
	return s.Map(func(element T) T { return element / value })
}

// Sqrt computes the square root elementwise.
func Sqrt[T Float](s Series[T]) Series[T] {
	return s.Map(func(value T) T { return T(math.Sqrt(float64(value))) })
}

// Exp computes e raised to each present value.
func Exp[T Float](s Series[T]) Series[T] {
	return s.Map(func(value T) T { return T(math.Exp(float64(value))) })
}

// Log computes the natural logarithm elementwise.
func Log[T Float](s Series[T]) Series[T] {
	return s.Map(func(value T) T { return T(math.Log(float64(value))) })
}

// Pow raises each present value to exponent.
func Pow[T Float](s Series[T], exponent T) Series[T] {
	return s.Map(func(value T) T { return T(math.Pow(float64(value), float64(exponent))) })
}

// Sum returns the sum of present values and whether any are present.
func Sum[T Number](s Series[T]) (T, bool) {
	return reduce.Sum[T](s, nil)
}

// Mean returns the arithmetic mean of present values and whether any are
// present.
func Mean[T Real](s Series[T]) (float64, bool) {
	return reduce.Mean[T](s, nil)
}

// Min returns the smallest present value and whether any are present. NaN
// propagation follows Go's min.
func Min[T cmp.Ordered](s Series[T]) (T, bool) {
	return reduce.Min[T](s, nil)
}

// Max returns the largest present value and whether any are present. NaN
// propagation follows Go's max.
func Max[T cmp.Ordered](s Series[T]) (T, bool) {
	return reduce.Max[T](s, nil)
}

// ArgMin returns the row index of the first minimum present value and whether
// any value is present. If a present value is NaN, it returns the first NaN
// row.
func ArgMin[T cmp.Ordered](s Series[T]) (int, bool) {
	index := 0
	var minimum T
	found := false
	for i, value := range s.values {
		if s.validity != nil && !s.validity[i] {
			continue
		}
		if !found || (value != value && minimum == minimum) || (minimum == minimum && value < minimum) {
			index = i
			minimum = value
			found = true
		}
	}
	return index, found
}

// ArgMax returns the row index of the first maximum present value and whether
// any value is present. If a present value is NaN, it returns the first NaN
// row.
func ArgMax[T cmp.Ordered](s Series[T]) (int, bool) {
	index := 0
	var maximum T
	found := false
	for i, value := range s.values {
		if s.validity != nil && !s.validity[i] {
			continue
		}
		if !found || (value != value && maximum == maximum) || (maximum == maximum && value > maximum) {
			index = i
			maximum = value
			found = true
		}
	}
	return index, found
}

// SampleVariance returns sample variance and whether at least two values are
// present.
func SampleVariance[T Real](s Series[T]) (float64, bool) {
	count := 0
	mean := 0.0
	sumSquares := 0.0
	for i, value := range s.values {
		if s.validity != nil && !s.validity[i] {
			continue
		}
		count++
		delta := float64(value) - mean
		mean += delta / float64(count)
		sumSquares += delta * (float64(value) - mean)
	}
	if count < 2 {
		return 0, false
	}
	return sumSquares / float64(count-1), true
}

// SampleStdDev returns sample standard deviation and whether at least two
// values are present.
func SampleStdDev[T Real](s Series[T]) (float64, bool) {
	variance, ok := SampleVariance(s)
	if !ok {
		return 0, false
	}
	return math.Sqrt(variance), true
}

// Quantile returns the q-quantile using the Hyndman-Fan type 7 method and
// whether any value is present. A present NaN produces NaN. Quantile panics
// when q is outside [0, 1] or is NaN.
func Quantile[T Real](s Series[T], q float64) (float64, bool) {
	if math.IsNaN(q) || q < 0 || q > 1 {
		panic("series: Quantile: q outside [0, 1]")
	}
	values := make([]float64, 0, s.Len()-s.NullCount())
	for i, value := range s.values {
		if s.validity != nil && !s.validity[i] {
			continue
		}
		converted := float64(value)
		if math.IsNaN(converted) {
			return math.NaN(), true
		}
		values = append(values, converted)
	}
	if len(values) == 0 {
		return 0, false
	}
	slices.Sort(values)
	position := float64(len(values)-1) * q
	lower := int(position)
	upper := min(lower+1, len(values)-1)
	fraction := position - float64(lower)
	if fraction == 0 || lower == upper {
		return values[lower], true
	}
	return values[lower]*(1-fraction) + values[upper]*fraction, true
}

// Median returns the 0.5 quantile and whether any value is present.
func Median[T Real](s Series[T]) (float64, bool) {
	return Quantile(s, 0.5)
}

// CumSum returns running sums. Null rows remain null and do not disturb the
// accumulator.
func CumSum[T Number](s Series[T]) Series[T] {
	var zero T
	return s.Scan(zero, func(sum, value T) T { return sum + value })
}

// Sorted returns a stable ascending ordering with nulls last.
func Sorted[T cmp.Ordered](s Series[T]) Series[T] {
	return s.SortedFunc(cmp.Compare)
}

// SortedDescending returns a stable descending ordering with nulls last.
func SortedDescending[T cmp.Ordered](s Series[T]) Series[T] {
	return s.SortedFunc(func(left, right T) int { return cmp.Compare(right, left) })
}

func matchRows[T any](a, b Series[T], lengthMismatch string, predicate func(T, T) bool) mask.Mask {
	if a.Len() != b.Len() {
		panic(lengthMismatch)
	}
	return mask.NewFunc(a.Len(), func(i int) bool {
		return (a.validity == nil || a.validity[i]) &&
			(b.validity == nil || b.validity[i]) &&
			predicate(a.values[i], b.values[i])
	})
}

func matchValue[T any](s Series[T], predicate func(T) bool) mask.Mask {
	return mask.NewFunc(s.Len(), func(i int) bool {
		return (s.validity == nil || s.validity[i]) && predicate(s.values[i])
	})
}
