# V56 current-state recognition repair

The read-only recognizer no longer minimizes a projected Frank--Bilby scalar
over both signs. Ray traversal declares the sign geometrically, the complete
three-vector mismatch (including transverse leakage) is evaluated, and the
largest residual over the sampled rays controls qualification. The explicit
counterexample `(1,10,0)` versus `(1,0,0)` now has zero projected error but a
full-vector relative residual of 10 and cannot qualify.

Connected support, centroid evaluation, erosion/dilation, and ray traversal
are periodic. Interior, shell, and circuit widths are supplied in metres and
their realized values are reported. Orientation averages use circular means.

This is a recognition repair only. It does not authorize physical promotion.
Two requirements remain:

1. choose and test the model's crystal-symmetry orientation equivalence rather
   than only 2-pi angle periodicity;
2. extend neutral handoff auditing from reconstructed bulk fields to complete
   interface/order energy and all retained history.

Until both close, a positive read-only candidate must remain diagnostic and
cannot allocate a new physical grain or launch unforced growth.
