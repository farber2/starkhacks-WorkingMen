# iPhone Meta-Ready Shell (SwiftUI)

This folder contains a minimal iPhone integration layer that reuses the existing FastAPI backend.

## What It Includes

- SwiftUI app shell with:
  - Meta glasses connection card
  - Capture Board action
  - Get Help action
  - Coach response display
  - Backend/camera/audio status panels
- FastAPI client for:
  - `GET /state`
  - `POST /help`
  - `POST /vision/board`
  - `POST /glasses/audio`
- Mock wearables services:
  - `WearablesConnectionService`
  - `WearablesCameraService`
  - `WearablesAudioService`

## Future Meta iOS SDK Integration Points

- `Services/WearablesCameraService.swift`
  - Replace mock capture with real frame ingestion from Meta wearables toolkit.
- `Services/WearablesAudioService.swift`
  - Replace mock route logic with real glasses audio output routing.
- `Services/WearablesConnectionService.swift`
  - Replace mock connect/disconnect with device discovery/session lifecycle.

## Notes

- Chess logic stays in Python backend.
- This iOS layer is intentionally lightweight and hackathon-friendly.

## Create A Real Runnable Xcode Project (5-10 min)

Use this once to create the missing `.xcodeproj` cleanly in Xcode.

1. Open Xcode -> `File` -> `New` -> `Project...`
2. Choose `iOS` -> `App` -> `Next`
3. Product Name: `ChessTutorIOS`
4. Team: your Apple ID team (or set later)
5. Organization Identifier: e.g. `com.yourname`
6. Interface: `SwiftUI`
7. Language: `Swift`
8. Testing System: default is fine
9. Save location: `/Users/dkplayz/Documents/code/Hackathon/starkhacks-WorkingMen/ios`
10. This creates `ios/ChessTutorIOS.xcodeproj` and a default app folder.

## Wire In Existing Source Files

Inside Xcode project navigator:

1. Delete default generated `ContentView.swift` and default `ChessTutorIOSApp.swift` (choose `Remove Reference`, not move to trash).
2. Right click the project root group -> `Add Files to "ChessTutorIOS"...`
3. Select these folders and files:
   - `ios/ChessTutorIOS/Views/ContentView.swift`
   - `ios/ChessTutorIOS/ViewModels/HomeViewModel.swift`
   - `ios/ChessTutorIOS/Services/APIClient.swift`
   - `ios/ChessTutorIOS/Services/WearablesConnectionService.swift`
   - `ios/ChessTutorIOS/Services/WearablesCameraService.swift`
   - `ios/ChessTutorIOS/Services/WearablesAudioService.swift`
   - `ios/ChessTutorIOS/Models/AppConfig.swift`
   - `ios/ChessTutorIOS/Models/APIModels.swift`
   - `ios/ChessTutorIOS/Models/WearablesModels.swift`
   - `ios/ChessTutorIOS/ChessTutorIOSApp.swift`
4. In the add dialog:
   - Check `Copy items if needed`: **off** (files are already in repo)
   - Check `Create groups`: **on**
   - Target `ChessTutorIOS`: **checked**

## Signing For Personal iPhone

1. Click the blue project icon -> select target `ChessTutorIOS`
2. Open `Signing & Capabilities`
3. Enable `Automatically manage signing`
4. Select your personal Apple ID Team
5. Ensure Bundle Identifier is unique, e.g. `com.yourname.ChessTutorIOS`
6. Connect iPhone, trust Mac, select your device in Xcode run target
7. First run may ask to trust developer certificate on iPhone:
   - `Settings` -> `General` -> `VPN & Device Management` -> trust your developer profile

## Backend URL For iPhone

`127.0.0.1` only works for simulator/local loopback.  
For physical iPhone, backend must use your Mac LAN IP.

In `ios/ChessTutorIOS/ChessTutorIOSApp.swift`, switch:

- from: `APIClient(config: .defaultLocal)`
- to: `APIClient(config: .defaultLAN)`

Then update `defaultLAN` IP in `AppConfig.swift` to your Mac IP.

## ATS (HTTP Local Dev) Setting

For plain `http://` local development, add this to your app `Info.plist`:

```xml
<key>NSAppTransportSecurity</key>
<dict>
    <key>NSAllowsArbitraryLoads</key>
    <true/>
</dict>
```

Hackathon-safe and simple.  
Later, tighten this to domain-specific exceptions.

## Run Checklist

1. Start backend on Mac:
   - `uvicorn backend.app:app --reload --host 0.0.0.0 --port 8000`
2. Confirm iPhone and Mac are on same Wi-Fi
3. Use correct LAN IP in `AppConfig.defaultLAN`
4. Build + Run from Xcode to iPhone
