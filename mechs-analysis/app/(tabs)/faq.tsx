// Updated SwingMetricsScreen.tsx

import { ScrollView, StyleSheet, View, Dimensions } from 'react-native';
import { ThemedText } from '@/components/ThemedText';
import { LinearGradient } from 'expo-linear-gradient';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

export default function SwingMetricsScreen() {
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
          <ThemedText type="title" style={styles.title}>Understanding Your Swing Metrics</ThemedText>

          <ThemedText style={styles.introText}>
            Every swing is unique, but combining data from multiple metrics helps us ensure you're generating power efficiently and building a swing that will get you far. These ranges have been gathered from trusted and renowned baseball outlets worldwide, compiled from years of research on swing efficiency, biomechanics, and athletic development. These metrics are some of the most important to track — they offer a reliable, objective view of your swing efficiency without relying on subjective advice that can vary between hitting coaches. They're widely agreed upon in the baseball community and give you a great foundation for evaluating your swing. Below are the key metrics we track, with guidance across all levels — from youth to professional athletes.
          </ThemedText>

          {metrics.map((metric, idx) => (
            <View key={idx} style={styles.metricBlock}>
              <ThemedText style={styles.metricTitle}>{metric.name}</ThemedText>
              <ThemedText style={styles.metricDesc}>{metric.description}</ThemedText>

              {metric.ageGroups.map((group, i) => (
                <View key={i}>
                  <ThemedText style={styles.ageNoteTitle}>{group.label}:</ThemedText>
                  <ThemedText style={styles.metricIdeal}>Typical Range: {group.ideal}</ThemedText>
                  <ThemedText style={styles.ageNote}>{group.note}</ThemedText>
                </View>
              ))}
            </View>
          ))}
        </View>
      </ScrollView>
    </View>
  );
}

const metrics = [
  {
    name: 'Hip-Shoulder Separation',
    description: 'Measures the rotational angle between the hips and shoulders at foot plant. A larger separation stores energy like a coil and contributes to power. Aaron Judge, for instance, gets around 35°-50° of separation each swing.',
    ageGroups: [
      {
        label: 'Youth (7-12)',
        ideal: '5°-20°',
        note: 'Use medicine ball throws and resistance band drills to build core control. Reinforce separation using slow-motion mirror drills.'
      },
      {
        label: 'Middle School (12-13)',
        ideal: '10°-25°',
        note: 'Introduce step-back swings and toe-tap drills to isolate hip and shoulder sequencing.'
      },
      {
        label: 'High School (14-18)',
        ideal: '20°-40°',
        note: 'Focus on building rotational strength with med-ball slams and hip-shoulder separation drills that emphasize delayed torso rotation.'
      },
      {
        label: 'College (18-22)',
        ideal: '25°-45°',
        note: 'Use high-speed video or sensors to fine-tune timing and max torque. Layer core plyometrics with swing reps for dynamic control.'
      },
      {
        label: 'Professional',
        ideal: '30°-50°',
        note: 'Refine load-to-launch mechanics through feel-based drills. Prioritize swing efficiency while maximizing stored energy.'
      },
    ],
  },
  {
    name: 'Attack Angle',
    description: 'The upward or downward angle of the bat at contact. A slight positive attack angle helps generate line drives and maximize exit velocity.',
    ageGroups: [
      {
        label: 'Youth (7-12)',
        ideal: '-5° to +15°',
        note: 'Use tee work and front toss to groove a flat-to-slightly-upward path. Focus on hitting line drives into the outfield.'
      },
      {
        label: 'Middle School (12-13)',
        ideal: '0° to +18°',
        note: 'Use short bat drills and one-hand swings to control barrel path and limit uppercutting tendencies.'
      },
      {
        label: 'High School (14-18)',
        ideal: '0° to +20°',
        note: 'Match bat path to pitch plane using machine work and swing plane trainers. Track ball flight for feedback.'
      },
      {
        label: 'College (18-22)',
        ideal: '+5° to +22°',
        note: 'Integrate tech (Blast, Rapsodo) to refine attack angle. Adjust plane based on pitch height and velocity.'
      },
      {
        label: 'Professional (22+)',
        ideal: '+5° to +25°',
        note: 'Use real-time feedback tools to adjust attack angle zone-by-zone. Refine approach based on scouting data and pitch trends.'
      },
    ],
  },
  {
    name: 'Peak Hand Speed',
    description: 'Peak hand speed refers to the highest velocity your hands reach during the swing, specifically measured at the bat’s handle. This metric is crucial because it reflects how quickly and efficiently a hitter can move the bat into the zone. Higher hand speed often translates to better bat control, increased power, and the ability to adjust to different pitch speeds. It\'s a key indicator of swing quickness and overall athletic explosiveness at the plate.',
    ageGroups: [
      {
        label: 'Youth (7-12)',
        ideal: '8-15 mph',
        note: 'Incorporate hand-speed challenges with wiffle balls and light bats.'
      },
      {
        label: 'Middle School (12-13)',
        ideal: '13-18 mph',
        note: 'Pair weighted bat swings with live reps to build quick twitch. Measure outcomes with bat sensors when possible.'
      },
      {
        label: 'High School (14-18)',
        ideal: '18-26 mph',
        note: 'Use overload/underload bat training cycles and resistance work. Track gains using consistent drill benchmarks.'
      },
      {
        label: 'College (18-22)',
        ideal: '22-28 mph',
        note: 'Maximize acceleration through rotational med-ball throws, wrist loading drills, and resisted swings.'
      },
      {
        label: 'Professional',
        ideal: '24-30+ mph',
        note: 'Use high-speed analysis to isolate inefficiencies. Layer swing constraint drills to optimize both speed and adjustability.'
      },
    ],
  },
  {
    name: 'Time to Contact',
    description: 'How quickly the swing travels from launch position to contact. A lower value suggests better efficiency and reactivity.',
    ageGroups: [
      {
        label: 'Youth (7-12)',
        ideal: '0.22-0.30 sec',
        note: 'Prioritize rhythm and smoothness using slow-motion drills. Use front toss to build repeatable load-to-launch patterns.'
      },
      {
        label: 'Middle School (12-13)',
        ideal: '0.20-0.28 sec',
        note: 'Add stride timing drills and reaction-based hitting games to train faster swing launch with control.'
      },
      {
        label: 'High School (14-18)',
        ideal: '0.15-0.22 sec',
        note: 'Introduce high-velocity machines and short-distance toss to develop quick trigger under pressure.'
      },
      {
        label: 'College (18-22)',
        ideal: '0.13-0.19 sec',
        note: 'Simulate real ABs using pitch recognition drills with variable timing. Record and review quickness frame-by-frame.'
      },
      {
        label: 'Professional',
        ideal: '0.12-0.18 sec',
        note: 'Refine compact swing mechanics under high velo. Use advanced tools like VR hitting and live pitch tracking.'
      },
    ],
  },
  {
    name: 'Lunging Detection',
    description: 'Evaluates whether the body drifts too far forward before contact, which hurts balance and power transfer.',
    ageGroups: [
      {
        label: 'Youth (7-12)',
        ideal: 'Minimal forward drift',
        note: 'Teach balance and centered weight — overstepping or falling forward is common at this stage.',
      },
      {
        label: 'Middle School (12-13)',
        ideal: 'Mostly stable center',
        note: 'Refine posture at front foot plant. Start using cues like “stay behind the ball.”',
      },
      {
        label: 'High School (14-18)',
        ideal: 'Stable hips/head before contact',
        note: 'Lunging may reflect pitch recognition flaws — work on tracking and loading rhythm.',
      },
      {
        label: 'College (18-22)',
        ideal: 'Late weight transfer',
        note: 'Swing should reflect advanced balance. Slight drift is acceptable if tied to aggressive torque.',
      },
      {
        label: 'Professional',
        ideal: 'Explosive but centered',
        note: 'Pros generate late power without early leak. Use video/tech to eliminate subtle lunging.',
      },
    ],
  },
];

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
  metricBlock: {
    marginBottom: 24,
    borderBottomWidth: 1,
    borderBottomColor: '#ffffff22',
    paddingBottom: 16,
  },
  metricTitle: {
    color: '#ffab91', // 🔶 warmer coral tone for contrast
    fontWeight: '800',
    fontSize: 18,      // ⬆️ from 16 to 18
    marginBottom: 8,   // extra breathing room
    textTransform: 'uppercase', // optional, for visual punch
  },
  metricDesc: {
    color: '#cfeadb',
    fontSize: 14,
    marginBottom: 6,
  },
  metricIdeal: {
    color: '#9fe3b4',
    fontSize: 13,
    fontStyle: 'italic',
    marginBottom: 4,
  },
  ageNoteTitle: {
    color: '#a9dacc',
    fontWeight: '600',
    fontSize: 13,
    marginTop: 8,
  },
  ageNote: {
    color: '#c2efe2',
    fontSize: 13,
    marginBottom: 6,
  },
});
