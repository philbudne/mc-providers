Media Cloud Providers Library
=============================

A package of search providers for Media Cloud, wrapping up interfaces for different social media platform.

UNDER CONSTRUCTION- Probably won't get a huge amount of attention for a little bit, but I'm putting this 
up now since I've done this extraction already.

Install with pip (`pip install .`) and the `install.sh` script. 

Requires environment variables set for various interfaces to work correctly.


### Build

Make sure `pip install flit twine` so you can build and deploy to PyPI.

1. Run `flit build` to create an installation package
2. Run `twine upload --repository-url https://test.pypi.org/legacy/ dist/*` to upload it to PyPI's test platform
3. Run `twine upload dist/*` to upload it to PyPI

### Version History
* __v0.2.0__ - add in support for Media Cloud legacy database
* __v0.1.7__ - corrected support for a "filters" kwarg in online_news
* __v0.1.6__ - Added support for a "filters" kwarg in online_news
* __v0.1.5__ - Added politeness wait to all chunked queries in twitter provider
* __v0.1.4__ - Added Query Chunking for large collections in the Twitter provider
* __v0.1.3__ - Added Query Chunking for large queries in the onlinenews provider
* __v0.1.2__ - Test Completeness
* __v0.1.1__ - Parity with web-search module, and language model
* __v0.1.0__ - Initial pypi upload 
* 