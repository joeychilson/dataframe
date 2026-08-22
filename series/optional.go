package series

// Optional carries a value that may be absent. Its zero value is None. It is
// used when a nullable cell must itself be stored or passed as one value;
// ordinary accessors and reductions use Go's comma-ok convention.
type Optional[T any] struct {
	// Value holds the optional value. It is meaningful only when Valid is true.
	Value T
	// Valid reports whether Value is present.
	Valid bool
}

// Some returns a present Optional containing value.
func Some[T any](value T) Optional[T] {
	return Optional[T]{Value: value, Valid: true}
}

// None returns an absent Optional.
func None[T any]() Optional[T] {
	return Optional[T]{}
}

// Get returns the value and whether it is present, following Go's comma-ok
// convention. An absent Optional returns the zero value of T and false.
func (o Optional[T]) Get() (T, bool) {
	if !o.Valid {
		var zero T
		return zero, false
	}
	return o.Value, true
}

// Or returns the present value or fallback when o is absent.
func (o Optional[T]) Or(fallback T) T {
	if !o.Valid {
		return fallback
	}
	return o.Value
}
