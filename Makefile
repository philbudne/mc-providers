# from es-tools

# to create development environment: `make install`
# to run pre-commit linting/formatting: `make lint`

VENVDIR=.venv
VENVBIN=$(VENVDIR)/bin
VENVDONE=$(VENVDIR)/.done
LID_BIN=mc_providers/language/lid.176.bin
LID_NEW=$(LID_BIN).new

help:
	@echo Usage:
	@echo "make install -- installs pre-commit hooks, dev environment"
	@echo "make lint -- runs pre-commit checks"
	@echo "make test-mc -- run MC news provider tests"
	@echo "make test[-all] -- run all tests"
	@echo "make update-pre-commit -- update .pre-commit-config.yaml"
	@echo "make clean -- remove development environment"

## run pre-commit checks on all files
lint:	$(VENVDONE) check-lid
	$(VENVBIN)/pre-commit run --all-files

# create venv with project dependencies
# pre-commit is in dev optional-requirements
# mypy run from dev .venv when using uv
install $(VENVDONE): Makefile pyproject.toml
	uv sync --extra dev --extra test --extra mypy --extra release
	$(VENVBIN)/pre-commit install
	touch $(VENVDONE)

check-lid:
	curl -SL https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin -o $(LID_NEW)
	cmp $(LID_NEW) $(LID_BIN) || cp -f $(LID_NEW) $(LID_BIN)

# tag and push (adds .latest tag):
release: lint
	$(VENVBIN)/python release.py release

## run by pre-commit:
## upgrade packages, need explicit --upgrade-package
## to pick up mediacloud private packages .latest tag changes
relock uv.lock: pyproject.toml
	uv lock --upgrade `awk '/git.https:\/\/github.com\/mediacloud\// { print }' pyproject.toml | sed -e 's/^ *"/--upgrade-package /' -e 's/ *@.*//'`

## update .pre-commit-config.yaml
update-pre-commit:	$(VENVDONE)
	$(VENVBIN)/pre-commit autoupdate

test test-all:	$(VENVDONE)
	@test -n "$$MEDIA_CLOUD_API_KEY" || (echo "need MEDIA_CLOUD_API_KEY" 1>&2 && exit 1)
	$(VENVBIN)/pytest

test-mc:	$(VENVDONE)
	@test -n "$$MEDIA_CLOUD_API_KEY" || (echo "need MEDIA_CLOUD_API_KEY" 1>&2 && exit 1)
	$(VENVBIN)/pytest -v mc_providers/test/test_onlinenews.py::OnlineNewsMediaCloudProviderTest

## clean up development environment
clean:
	-$(VENVBIN)/pre-commit clean
	rm -rf $(VENVDIR) build *.egg-info .pre-commit-run.sh.log \
		__pycache__ .mypy_cache
