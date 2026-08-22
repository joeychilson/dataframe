// Package mask provides immutable, bit-packed row selections.
//
// Masks are deliberately independent of dataframe and series so both packages
// can consume them without an import cycle. Series comparisons produce Mask
// values, and series.AsMask converts nullable boolean values to a Mask.
package mask
