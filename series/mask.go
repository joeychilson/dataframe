package series

import "github.com/joeychilson/dataframe/mask"

// AsMask converts nullable boolean values to a two-valued Mask. Present true
// values select their rows; false and null values do not. The returned Mask
// has the same length as values.
func AsMask(values Series[bool]) mask.Mask {
	if values.validity == nil {
		return mask.New(values.values)
	}
	nullCount := values.NullCount()
	if nullCount == 0 {
		return mask.New(values.values)
	}
	if nullCount == values.Len() {
		return mask.None(values.Len())
	}
	return mask.NewFunc(values.Len(), func(i int) bool {
		return values.values[i] && values.validity[i]
	})
}
