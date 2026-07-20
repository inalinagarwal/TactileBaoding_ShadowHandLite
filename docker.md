

## 🖐️ Running the ShadowLite Hand (with TouchLab sensors)

### Why this doc exists

The main reason I’m writing this is because ***THIS SETUP IS A PAIN***.

Yes, I took the time to bold *and* italicize that.

If you’re a Docker pro, I am jealous of you. This whole setup took way longer than it should have, mostly because getting everything to run properly inside Docker is… not fun. So this is basically a “save your future self (and others) from pain” guide.

---

### Step 1: Do the official setup first

Before anything, go through the official guide:

[[https://shadow-robot-company-dexterous-hand-lite.readthedocs-hosted.com/en/latest/](https://shadow-robot-company-dexterous-hand-lite.readthedocs-hosted.com/en/latest/)](https://shadow-robot-company-dexterous-hand-lite.readthedocs-hosted.com/en/latest/](https://shadow-robot-company-dexterous-hand-lite.readthedocs-hosted.com/en/latest/))

---

### Step 2: Launch the hand

Once everything is set up, open the Shadow Robot launcher, click **“Dexterous Hand”**, then launch **“Shadow Right Hand. Desktop”**

If things are working, four terminal windows should open along with RViz. RViz might take a while to load, so don’t immediately assume something is broken (this note is for me lol).

### Stuff we learned the hard way (please read 😭)

First — Ethernet Adapters. This one is actually cursed. We have four hands in the lab, and each one needs to be connected to the correct Ethernet adapter. If you mix them up, the robot just won’t work. No clear errors, no helpful messages, it will just say "can't read". We lost about four hours to this, so double check before you start questioning everything.

Second — RViz is slow. Like, really slow sometimes. That’s normal. The better way to check if things are working is to look at the terminals. If you see *`dexterous_hand`** show up in one of them, that’s your sign that everything is actually running fine. RViz will catch up eventually.

## 🐳 Step 3: Docker (aka the boss fight)

So the goal here is simple (in theory):

you want a Docker container running ROS, with the Shadow hand and TouchLab sensors working inside it.

In reality… yeah good luck 👍

First thing: you need to create and run a container.

There are technically two ways to do this. One of them sucks, and I have chosen peace in life by ignoring it.

The bad way is manually building a Docker image and installing everything yourself. I was actually doing this for a while… and my god it nearly sent me insane. Save yourself. Don’t do it.

### The way that actually works

Run this to create the container:

```bash

docker run -d --name shadow-hand-test \

  --privileged \

  --network host \

  -v ~/shadow_docker_ws:/root/shadow_docker_ws \

  -v ~/repos:/root/repos \

  shadowrobot/dexterous-hand:noetic-night-build \

  tail -f /dev/null

```

This basically spins up a container called *`shadow-hand-test`** with all the right permissions and mounts so ROS + the hand can actually function.

### Start and enter the container

Once the container is created, you don’t need to recreate it every time.

Just start it:

```bash

docker start shadow-hand-test

```

Then jump inside:

```bash

docker exec -it shadow-hand-test bash

```

That command drops you into the container, and now you’re living inside Docker whether you like it or not.

### Open it in VS Code (seriously, do this)

Now open VS Code and use the **Dev Containers** extension to attach to the container.

Trust me on this — this will save you *a lot* of pain. Trying to do everything through the terminal alone gets messy fast.

Once you attach, VS Code basically treats the container like a normal dev environment, which makes life way easier.

---

### Important thing that will mess you up if you ignore it

Inside the container, your file structure actually matters.

Anything you’re working on needs to go in one of these (well not anything, this is all I know and needed lol):

* `/root/shadow_docker_ws/src`

* `/root/repos`

If you put stuff somewhere random, ROS will just pretend it doesn’t exist and you’ll sit there wondering why nothing is working.

### Where your code actually goes (this matters)

Inside the container, there are two main places you’ll be using:

* `/root/repos` → this is for any external repos, libraries, or dependencies your ROS stuff needs

* `/root/shadow_docker_ws/src` → this is where your actual ROS packages and scripts live

Think of it like:

* **repos = support stuff**

* **src = your actual ROS code**

If you mix this up, things won’t necessarily crash… they’ll just quietly not work, which is worse.

---

### ⚠️ Source. Everything. Every time.

Every single time you go inside the container, you need to source your environments.

Yes, every time. No, it won’t magically remember. A man can only wish.

Run these:

```bash

source ~/shadow_docker_ws/devel/setup.bash

source /root/shadow_docker_ws/devel/setup.bash

source /opt/ros/noetic/setup.bash

```

Keep these somewhere you can copy-paste because you *will* forget at some point and then wonder why ROS is acting weird.

---

### Running ROS stuff

Once everything is sourced properly, ROS should behave normally and you should be able to see topics and all that good stuff.

If you’re running a Python node, it’s just the usual:

```bash

rosrun my_shadow_control my_policy_[node.py](http://node.py)

```

Where:

* `my_shadow_control` = your ROS package

* `my_policy_node.py` = your script

if it’s not working, 90% chance something wasn’t sourced (again…).

## ⚠️ VERY IMPORTANT (read this before running your script)

Before you run any ROS script, **check the control mode**.

By default, the Shadow hand is set to **trajectory control**.

If your script expects something else (like position control), it will just… not work.

To change it:

* Go to one of the original terminals that opened with the hand (don’t open a new one)

* You should see something like `[user@server]─[~]`

* Run `rqt`

* Then go to:

  **Plugins → Shadow Robot → Control Mode**

From there, switch to whatever your script needs (e.g. position control).

If you don’t do this, your script will fail and give you absolutely no helpful explanation 👍

## Step 4: Where is TouchLab?? (the confusing part)

If you made it this far — first of all, yay 🎉

Now you’ll probably look at the ROS topics and think:

**“where is TouchLab?”**

Yeah… good question.

This part is a bit confusing and honestly I don’t fully know what magic is happening behind the scenes, but it *does* work.

The TouchLab sensors don’t show up in a super obvious way like you’d expect, but they are there and integrated into the system once everything is running properly.

So if:

* your container is running

* everything is sourced

* the hand is launched

* and you’re seeing normal topics

then you’re probably fine.

(If not… welcome back to debugging 😭)

## 🧠 TouchLab Sensors (finally)

Before anything — physically connect the **sensors computer** to the **Shadow server laptop**.

I’ll show two ways to use TouchLab:

1. Using PlotJuggler (to see the data)

2. Using it inside your ROS scripts

---

### Method 1: PlotJuggler (aka “is this thing even working?”)

Go into VS Code and attach to the touch lab Docker container (already setup by touchlab - This is where we take 2 mins to appreciate touchlab). 

Now inside the container, run:

```bash id="6p3h9x"

roslaunch touchlab_driver_ros touchlab_driver.launch calibration:=/ros1/calibration/uoe-default.bin

```

What this does is continuously publish all the sensor data.

Now open another terminal (still inside the container) and run:

```bash id="r2k8mz"

rosrun plotjuggler plotjuggler

```

This opens PlotJuggler.

And yes — **a meme will pop up first**.

Caught me off guard too. Whoever put that there is a legend.

### Understanding the data (so you don’t get confused)

When you start looking at the topics, you’ll notice two main ones:

* `touchlab_driver_raw`

* `touchlab_driver_calibrated`

The **raw topic** is pretty straightforward — you can clearly see all **64 sensors** updating, so it’s easy to understand what’s going on.

The **calibrated topic** is where things get a bit confusing.

Some of the values won’t immediately make sense, but the key thing to remember is:

> **every 3rd value corresponds to one taxel**

So when you’re subscribing to calibrated data, don’t just assume each value is one sensor. The data is grouped, and you need to interpret it accordingly.

If you don’t keep this in mind, you’ll probably misread the data and wonder why things look off.

---

### Seeing the data

Once PlotJuggler opens:

* Look at the left panel

 *You should see* *“Streaming”**

 *Select* *ROS Topic Subscriber**

 *Hit* *Start**

Now pick the topic you want:

* `touchlab_driver_calibrated`

* or `touchlab_driver_raw`

Once selected, the data should appear under the **time series list**.

### The fun part (actually seeing it work)

Here’s the cool bit:

You can **drag any data field** from the list onto the plot, and it’ll start visualising in real time.

Now press on the sensors with your fingers and you’ll literally see the values change live.

Each spike = you touching a taxel.

It’s actually really satisfying and also the easiest way to confirm everything is working properly.

---

### If you want the exact ordering

If you’re trying to properly understand how the calibrated data is structured (instead of guessing like the rest of us), there’s actual documentation for it:

👉 [[https://touchlab.atlassian.net/wiki/spaces/KB/overview](https://touchlab.atlassian.net/wiki/spaces/KB/overview)](https://touchlab.atlassian.net/wiki/spaces/KB/overview](https://touchlab.atlassian.net/wiki/spaces/KB/overview))

This explains how the data is ordered and how each value maps to the sensors/taxels.

So if you’re doing anything serious with the data (like control policies or learning), you’ll probably want to look at that page instead of guessing the mapping like me.

## 🚀 Method 2: Using TouchLab in your ROS scripts (the actual important part)

Alright, this is the part that actually matters if you want to *use* the sensors.

What you need to do is take the TouchLab ROS packages (the two folders inside the TouchLab container’s `src`) and copy them into your Shadow container.

So basically:

 *go into the* *TouchLab container**

* copy the two folders from its `src` which are `touchlab_driver_ros` and `touchlab_msgs`

* paste them into:

  `/root/shadow_docker_ws/src` inside your **shadow-hand container**

---

### Build it

Now inside the Shadow container, run:

```bash id="x7k2qp"

cd /root/shadow_docker_ws/

catkin_make

source devel/setup.bash

```

And… that’s it.

No really, that’s the step.

---

### Important note (this confused me a LOT)

When you use the PlotJuggler method earlier, you *can* see the TouchLab topics in the shadow container.

And if you check topic lists across containers, you’ll notice:

* the topics exist

* they show up

BUT…

> you won’t actually be able to `rostopic echo` them properly from the Shadow container

There *is* a reason for this (something to do with how things are being published across setups), but I honestly forgot the exact explanation 💀

Now you can run any ROS script and Subscribe to jointstates and Touchlab sensor data inside the Docker Container!!!

🎉 You’re done (for real this time)