#!/usr/bin/env bash

# Script to run linting tests.

action() {
    local this_file="$( [ ! -z "$ZSH_VERSION" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "$this_file" )" && pwd )"
    local repo_dir="$( cd "$( dirname "$this_dir" )" && pwd )"

    (
        cd "$repo_dir"
        flake8 cmsml tests setup.py
    )
}
action "$@"
