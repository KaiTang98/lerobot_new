First follow then steps from [README.md](../README.md) in the repository root. Then install the packages from *requirements.txt* and continue the introduction in [README.dm](../README.md) in the root folder to run the code.

## Scripts description

The script [reader.py](reader.py) will automatically install the APK to the connected Oculus Quest and start program. By default it uses the APK delivered with the repository. If you decide to compile your own version of the app with the code from *app_source* folder, please adjust the `APK_path` argument when running `OculusReader.install` function.

Use script [install.py](install.py) to reinstall or uninstall the APK from the connected Oculus Quest.

## Stopping the app

You can stop the teleoperation app on the headset in two ways:

1) Directly via adb:

```
adb shell am force-stop com.rail.oculus.teleop
```

2) Using the helper script:

```
python -m lerobot.teleoperators.quest.oculus_reader.install --stop
```

The `OculusReader.stop()` method now also attempts to force-stop the APK automatically when you shut down the reader.
