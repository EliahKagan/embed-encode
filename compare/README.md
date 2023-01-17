Comparison of float64-valued embeddings from Python and Java.

Note that this only compares them for equality to double precision. That should
be sufficient, but potentially they could be parsed to BigDecimal and compared.

Note also that the model is not deterministic. The JSON files checked into the
repository happen to have the same float64 values because they were from
requests in which the OpenAI embeddings API endpoint happened to process them
the same way. The point here is only to show how the language one uses doesn't
affect the results, if one is careful. But other factors, which vary across
separate runs even using the same exact client code, do affect them.
