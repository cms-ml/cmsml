#!/usr/bin/env bash

action() {
    local this_file="$( [ ! -z "$ZSH_VERSION" ] && echo "${(%):-%x}" || echo "${BASH_SOURCE[0]}" )"
    local this_dir="$( cd "$( dirname "$this_file" )" && pwd )"

    build_image() {
        local docker_file="$1"
        local image="$2"
        local tag="$3"


        if [ ! -f "$this_dir/$docker_file" ]; then
            2>&1 echo "docker file $docker_file not found in $this_dir"
            return "2"
        fi

        (
            echo "building image cmsml/$image:$tag from $this_dir/$docker_file"
            cd "$this_dir"
            docker build --file "$docker_file" --tag "cmsml/$image:$tag" .
        )
    }

    tag_and_push() {
        local image="$1"
        local base_tag="$2"

        # tag
        for tag in "${@:3}"; do
            docker tag "cmsml/$image:$base_tag" "cmsml/$image:$tag"
        done

        # push
        if [ "$CMSML_PUSH_IMAGE" != "0" ]; then
            for tag in "${@:2}"; do
                docker push "cmsml/$image:$tag"
            done
        fi
    }

    # build, tag and push all images
    build_image Dockerfile_27 cmsml 2.7
    tag_and_push cmsml 2.7 2

    build_image Dockerfile_37 cmsml 3.7
    tag_and_push cmsml 3.7

    build_image Dockerfile_38 cmsml 3.8
    tag_and_push cmsml 3.8 3 latest

    build_image Dockerfile_docs docs latest
    tag_and_push docs latest
}

# entry point
action
