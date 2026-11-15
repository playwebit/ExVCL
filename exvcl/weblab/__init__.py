"""exvcl.weblab -- a small, self-contained web-attack lab for testing ExVCL end to end on a VM.

A deliberately vulnerable file viewer (path traversal) is executed ONLY inside a sandbox worker under strace: the evidence of a request is
what its execution really does. An independent text model (character n-gram TF-IDF + logistic regression) labels the requests; ExVCL
decides which labelled requests may update a continual learner. Nothing here is exposed to a network: the worker opens a path and reports
only whether it exists and its size. Intended for an isolated lab VM.
"""
