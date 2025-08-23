import { View, StyleSheet } from 'react-native';
import { ThemedText } from '@/components/ThemedText';

export default function SwingsScreen() {
  return (
    <View style={styles.container}>
      <ThemedText type="title" style={styles.title}>
        Recent Swings
      </ThemedText>
      <ThemedText style={styles.subtitle}>
        Your swing analysis history will appear here
      </ThemedText>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#082419', // matching your login screen background
    padding: 24,
    justifyContent: 'center',
    alignItems: 'center',
  },
  title: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#ffffff',
    marginBottom: 12,
    textAlign: 'center',
  },
  subtitle: {
    fontSize: 16,
    color: '#d5ead6',
    textAlign: 'center',
  },
});