# VIM + TMUX + SSH

**Versions**
VIM     -- 
TMUX    --


## Configuring Copy and Paste

### Install the correct version of VIM

For copy-paste to work you need a version of VIM that supports this. The native version of vim on most machines does not support this by default

Check you version of vim with `vim -V`

```
+clipboard
-xterm_clipboard
```

If you do not have the correct version of vim, i.e. any version that shows `-clipboard` indicating that clipboard is not supported, you will have to install the appropriate version. `vim-gtk` is a good choice.

```
$ sudo apt install vim-gtk
```

### Configure VIM to use the system clipboard

Set he default registers in vim to access the systems clipboard. These corresponding to register `*` and `+`. By default, these are set to ?.

In your .vimrc, include the line: 

~~~bash
set clipboard^=unnamed,unnamedplus
~~~


### Establish a secure x11 connection with the server

1. When you're sshing from a mac, you need to have xquarts install. This can be done with [homebrew]().

2. Enabled trusted X11 forwarding on the client

On the server machine, edit the file `~/.ssh/sshd_config`

3. If you're connecting to a trusted source, use the `-Y` flag over the `-X` flag.

```
ssh -Y username@ip_address
```

5. As a sanity check to convince your self the connection is working, run the command `xeyes`. If a window does not popup with a little interactive animation of eyes, the X11 tunnel was not established.

### Configure TMUX to use the system clipboard


### Trouble shooting 

You did the above steps and copy+paste is still not working?

#### Ensure `DISPLAY` is correctly set

If you detach then reattach your tmux session and you copy paste stops working, it might be an issue in how the `DISPLAY` environment variable is being set. To trouble shoot this, in your tmux session type `echo $DISPLAY` it should return something like `localhost:XX.0`. For the sake of an example, lets say it returns `localhost:10.0`

Now, detach your tumx session and query `DISPLAY` again, if you get a different response, for example `localhose:11.0` then there is an issue with tmux not sending the information through the X11 tunnel using the correct loop-back address. 

This is an easy fix. Reattach tmux and set the `DISPLAY` variable to be the same as then tmux is detached. 

```
export DISPLAY=localhost:11.0
```

If this is happening regularly to you, check out this useful code snipped by ?.


#### Disable the mouse

For reasons I don't understand, enabling the interactive mouse can interfere with the copy paste

```
set mouse=r
```

#### `unnamed` and `unnamedplus` needs to be *prepended* to the clipboard.

Ensure that you've *prepended* `unnamed` and `unnamedplus` to the clipboard

```
set clipboard=unnamed,unnamedplus       # BAD
set clipboard+=unnamed,unnamedplus      # BAD
set clipboard^=unnamed,unnamedplus      # GOOD
```
