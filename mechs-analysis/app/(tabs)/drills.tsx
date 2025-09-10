'use client';

import { ScrollView, Text, View, StyleSheet, Dimensions } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

const allDrills = [
  'Stop on Contact',
  'Knob to the Pitcher',
  'Connection Ball Full Swing',
  'PVC Plane Drill',
  'Mini Wiffle Balls',
  'PVC Pipe',
  'Med Ball Drill',
  'Pause Drill',
  'Med Ball Throws',
  'Russian Twists',
  'Walkthrough Drill',
  'Heavier Bat to Train Speed',
  'On Knee Drill',
  'Soft Toss Visual Recognition Drill',
  'Pause and Wait Drill',
  'No Stride Drill',
  'Limited Stride Drill',
  'Wide Stance Drill',
];

export default function DrillsPage() {
  return (
    <View style={styles.container}>
      <View style={styles.bg} pointerEvents="none">
        <LinearGradient
          colors={["#0f1f18", "#123224", "#0f1f18"]}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={StyleSheet.absoluteFillObject}
        />
      </View>

      <ScrollView contentContainerStyle={styles.scrollContent} showsVerticalScrollIndicator={false}>
        <View style={styles.card}>
          <Text style={styles.title}>Swing Training Drills</Text>

          <Text style={styles.introText}>
            These proven drills target specific swing mechanics to help you improve your performance across all key metrics. Each drill is designed to address common issues and build muscle memory for optimal swing patterns. Practice these consistently to see measurable improvements in your swing efficiency and power generation.
          </Text>

          {allDrills.map((drill, index) => (
            <View key={index} style={styles.drillItem}>
              <Text style={styles.drillNumber}>{index + 1}</Text>
              <Text style={styles.drillName}>{drill}</Text>
            </View>
          ))}
        </View>
      </ScrollView>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#082419',
  },
  bg: {
    ...StyleSheet.absoluteFillObject,
    zIndex: -1,
  },
  scrollContent: {
    paddingBottom: 40,
    paddingHorizontal: 24,
    paddingTop: 40,
    flexGrow: 1,
    justifyContent: 'center',
  },
  card: {
    width: SCREEN_WIDTH < 400 ? '100%' : 360,
    backgroundColor: '#1a2e25',
    borderRadius: 20,
    padding: 24,
    borderWidth: 2,
    borderColor: '#ffffff33',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.2,
    shadowRadius: 20,
    elevation: 6,
    alignSelf: 'center',
  },
  title: {
    fontSize: 26,
    fontWeight: '900',
    color: '#ffffff',
    letterSpacing: 1,
    textAlign: 'center',
    marginBottom: 20,
    textShadowColor: '#f44336',
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 10,
  },
  introText: {
    color: '#d5ead6',
    fontSize: 15,
    marginBottom: 20,
    textAlign: 'center',
    lineHeight: 22,
  },
  drillItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 10,
    paddingLeft: 8,
  },
  drillNumber: {
    color: '#9fe3b4',
    fontSize: 14,
    fontWeight: '700',
    width: 24,
    marginRight: 12,
  },
  drillName: {
    color: '#c2efe2',
    fontSize: 14,
    flex: 1,
  },
});
