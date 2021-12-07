## Adding your runtime

This repository is a valid submission (and submission structure). 
You can simply add your dependencies on top of this repository.

Few of the most common ways are as follows:

* `requirements.txt` -- The `pip3` packages used by your inference code. As you add new pip3 packages to your inference procedure either manually add them to `requirements.txt` or if your software runtime is simple, perform:
    ```
    # Put ALL of the current pip3 packages on your system in the submission
    >> pip3 freeze >> requirements.txt
    >> cat requirementst.txt
    aicrowd_api
    coloredlogs
    matplotlib
    pandas
    [...]
    ```

* `apt.txt` -- The Debian packages (via aptitude) used by your inference code!

These files are used to construct your **AIcrowd submission docker containers** in which your code will run. 

* `Dockerfile` -- **For advanced users only**. `Dockerfile` gives you more flexibility on defining the software runtime used during evaluations. Please note that you should only use `aicrowd/learn-to-race:base` as the base image in your `Dockerfile`. If you have a use case that needs to use a different base image, please get in touch with us.

----

👋 In case you have any doubts or need help, you can reach out to us via Challenge [Discussions](https://www.aicrowd.com/challenges/learn-to-race-autonomous-racing-virtual-challenge/discussion) or [Discord](https://discord.gg/fNRrSvZkry).