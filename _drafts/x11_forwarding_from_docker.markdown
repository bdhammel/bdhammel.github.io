---
layout: post
title:  "X11 Forwarding from Docker"
---

To establish enable X11 forwarding from a docker container over ssh. the correct authentication credentials need to be mounted in the container.

This means you need to mount `.Xauthentication` and set the `hostname` of the container appropriately, such that it matched the authentication credentials.

Additionally, you need to ensure the `DISPLAY` environment variable matched the assigned DISPLAY on the host.

```
docker run --rm -it Dockerfile \
  -v
```
