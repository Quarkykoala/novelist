import './index.css';
import OceanMap from './components/OceanMap';

/**
 * Polymer-X: Zero-Cost Bioremediation Command Center
 * 
 * Main entry point for the web UI.
 * 
 * To enable Google Maps 3D Tiles:
 * 1. Get an API key from Google Cloud Console
 * 2. Enable Maps JavaScript API and Map Tiles API
 * 3. Pass the key to OceanMap component
 * 
 * Without an API key, a fallback ocean gradient is displayed.
 */

function App() {
  // Optional: Add your Google Maps API key here
  // const GOOGLE_MAPS_API_KEY = 'YOUR_API_KEY';

  return (
    <div className="w-full h-full bg-[#0a0a1a]">
      <OceanMap
      // apiKey={GOOGLE_MAPS_API_KEY}
      />

      {/* Version Badge */}
      <div className="fixed bottom-4 right-4 glass rounded-lg px-3 py-1.5 text-xs text-gray-400">
        POLYMER-X v0.2 • Committee Mode
      </div>
    </div>
  );
}

export default App;
