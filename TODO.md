# Erik's working notes (not to be modified by Claude)

- (CUDA) C++ algorithm

- algorithm for checking and/or correcting F

- algorithm for converting F to U. boundaries? logarithmic indexing?

- single precision?

- log2 instead of log or log10?

- texture hardware?

- assume H100

- limiting, projecting, atmosphere handling

- energy shift

- try with good ol' LS220? SRO?

- what if starting from EOS code (not table)? or high-res F table?

- generic "check state" and "project to valid state" functions, to be
  called after initial conditions, prolongating, reconstructing at
  cell faces, etc. (some work on cons, other on prims.)

- `using real = double` should imply that `real` is used everywhere, not `double`. (watch literal constants.)

- use only F from F table
