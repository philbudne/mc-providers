# mc-providers

A package of search providers for mediacloud, wrapping up interfaces for different social media platform.
UNDER CONSTRUCTION- Probably won't get a huge amount of attention for a little bit, but I'm putting this up now since I've done this extraction already.


Install with pip and the install script. 

Requires environment variables set for various interfaces to work correctly.


### Build


1. Run `flit build` to create an installation package
2. Run `twine upload --repository-url https://test.pypi.org/legacy/ dist/*` to upload it to PyPI's test platform
3. Run `twine upload dist/*` to upload it to PyPI



### Version History
* __v0.1.3__ - Added Query Chunking for large queries in the onlinenews provider
* __v0.1.2__ - Test Completeness
* __v0.1.1__ - Pairity with web-search module, and language model.
* __v0.1.0__ - Initial pypi upload. 