
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import React, { useState } from 'react';
import { Dimensions, ScrollView, StyleSheet, View } from 'react-native';

const { width } = Dimensions.get('window');

export default function AnalysisScreen() {
  const [selectedAnalysis, setSelectedAnalysis] = useState<number>(0);

  const analyses = [
    {
      id: 1,
      type: 'swing',
      date: '2 hours ago',
      score: 8.5,
      strengths: ['Good stance', 'Proper grip', 'Follow-through'],
      improvements: ['Hip rotation timing', 'Eye contact duration'],
      drills: ['Tee work for contact', 'Hip rotation exercises', 'Vision tracking drills']
    },
    {
      id: 2,
      type: 'pitch',
      date: '1 day ago',
      score: 7.8,
      strengths: ['Arm slot consistency', 'Good balance'],
      improvements: ['Stride length', 'Release point consistency'],
      drills: ['Long toss program', 'Balance beam work', 'Wall pitching drills']
    }
  ];

  const currentAnalysis = analyses[selectedAnalysis];

  return (
    <ScrollView style={styles.container}>
      <ThemedView style={styles.header}>
        <ThemedText type="title">Analysis Results</ThemedText>
        <ThemedText style={styles.subtitle}>
          AI-powered breakdown of your technique
        </ThemedText>
      </ThemedView>

      <ThemedView style={styles.analysisSelector}>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.selectorScroll}>
          {analyses.map((analysis, index) => (
            <View
              key={analysis.id}
              style={[
                styles.analysisTab,
                selectedAnalysis === index && styles.selectedTab
              ]}
            >
              <ThemedText
                style={[
                  styles.tabText,
                  selectedAnalysis === index && styles.selectedTabText
                ]}
                onPress={() => setSelectedAnalysis(index)}
              >
                {analysis.type === 'swing' ? '🏏' : '⚾'} {analysis.type.charAt(0).toUpperCase() + analysis.type.slice(1)}
              </ThemedText>
              <ThemedText style={styles.tabDate}>{analysis.date}</ThemedText>
            </View>
          ))}
        </ScrollView>
      </ThemedView>

      <ThemedView style={styles.scoreContainer}>
        <ThemedText type="subtitle" style={styles.sectionTitle}>Overall Score</ThemedText>
        <View style={styles.scoreCircle}>
          <ThemedText style={styles.scoreNumber}>{currentAnalysis.score}</ThemedText>
          <ThemedText style={styles.scoreMax}>/10</ThemedText>
        </View>
        <ThemedText style={styles.scoreDescription}>
          {currentAnalysis.score >= 8.5 ? 'Excellent technique!' : 
           currentAnalysis.score >= 7 ? 'Good form with room for improvement' : 
           'Focus on fundamentals'}
        </ThemedText>
      </ThemedView>

      <ThemedView style={styles.analysisSection}>
        <ThemedText type="subtitle" style={styles.sectionTitle}>✅ Strengths</ThemedText>
        <View style={styles.itemsContainer}>
          {currentAnalysis.strengths.map((strength, index) => (
            <View key={index} style={styles.strengthItem}>
              <ThemedText style={styles.itemText}>{strength}</ThemedText>
            </View>
          ))}
        </View>
      </ThemedView>

      <ThemedView style={styles.analysisSection}>
        <ThemedText type="subtitle" style={styles.sectionTitle}>🎯 Areas for Improvement</ThemedText>
        <View style={styles.itemsContainer}>
          {currentAnalysis.improvements.map((improvement, index) => (
            <View key={index} style={styles.improvementItem}>
              <ThemedText style={styles.itemText}>{improvement}</ThemedText>
            </View>
          ))}
        </View>
      </ThemedView>

      <ThemedView style={styles.analysisSection}>
        <ThemedText type="subtitle" style={styles.sectionTitle}>🏋️ Recommended Drills</ThemedText>
        <View style={styles.drillsContainer}>
          {currentAnalysis.drills.map((drill, index) => (
            <View key={index} style={styles.drillCard}>
              <ThemedText type="defaultSemiBold" style={styles.drillTitle}>
                Drill {index + 1}
              </ThemedText>
              <ThemedText style={styles.drillText}>{drill}</ThemedText>
              <View style={styles.drillMeta}>
                <ThemedText style={styles.drillDuration}>⏱️ 15-20 min</ThemedText>
                <ThemedText style={styles.drillFrequency}>🔄 3x per week</ThemedText>
              </View>
            </View>
          ))}
        </View>
      </ThemedView>

      <ThemedView style={styles.mechanicsSection}>
        <ThemedText type="subtitle" style={styles.sectionTitle}>⚙️ Detailed Mechanics</ThemedText>
        
        <View style={styles.mechanicsGrid}>
          <View style={styles.mechanicsCard}>
            <ThemedText style={styles.mechanicsTitle}>Stance</ThemedText>
            <ThemedText style={styles.mechanicsScore}>9.2/10</ThemedText>
            <ThemedText style={styles.mechanicsNote}>Excellent balance</ThemedText>
          </View>
          
          <View style={styles.mechanicsCard}>
            <ThemedText style={styles.mechanicsTitle}>Timing</ThemedText>
            <ThemedText style={styles.mechanicsScore}>7.8/10</ThemedText>
            <ThemedText style={styles.mechanicsNote}>Slightly early</ThemedText>
          </View>
          
          <View style={styles.mechanicsCard}>
            <ThemedText style={styles.mechanicsTitle}>Contact</ThemedText>
            <ThemedText style={styles.mechanicsScore}>8.5/10</ThemedText>
            <ThemedText style={styles.mechanicsNote}>Good extension</ThemedText>
          </View>
          
          <View style={styles.mechanicsCard}>
            <ThemedText style={styles.mechanicsTitle}>Follow Through</ThemedText>
            <ThemedText style={styles.mechanicsScore}>8.9/10</ThemedText>
            <ThemedText style={styles.mechanicsNote}>Complete rotation</ThemedText>
          </View>
        </View>
      </ThemedView>

      <ThemedView style={styles.actionButtons}>
        <View style={styles.primaryButton}>
          <ThemedText style={styles.primaryButtonText}>
            Save Analysis
          </ThemedText>
        </View>
        
        <View style={styles.secondaryButton}>
          <ThemedText style={styles.secondaryButtonText}>
            Share with Coach
          </ThemedText>
        </View>
      </ThemedView>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    padding: 20,
  },
  header: {
    marginTop: 40,
    marginBottom: 30,
    alignItems: 'center',
  },
  subtitle: {
    textAlign: 'center',
    opacity: 0.7,
    marginTop: 8,
    paddingHorizontal: 20,
  },
  analysisSelector: {
    marginBottom: 30,
  },
  selectorScroll: {
    flexDirection: 'row',
  },
  analysisTab: {
    backgroundColor: '#f9f9f9',
    padding: 15,
    borderRadius: 10,
    marginRight: 10,
    minWidth: 120,
    alignItems: 'center',
  },
  selectedTab: {
    backgroundColor: '#007AFF',
  },
  tabText: {
    fontWeight: 'bold',
    color: '#333',
  },
  selectedTabText: {
    color: 'white',
  },
  tabDate: {
    fontSize: 12,
    opacity: 0.7,
    marginTop: 4,
  },
  scoreContainer: {
    alignItems: 'center',
    backgroundColor: '#f9f9f9',
    padding: 30,
    borderRadius: 15,
    marginBottom: 30,
  },
  sectionTitle: {
    marginBottom: 15,
  },
  scoreCircle: {
    width: 120,
    height: 120,
    borderRadius: 60,
    backgroundColor: '#007AFF',
    justifyContent: 'center',
    alignItems: 'center',
    marginBottom: 15,
  },
  scoreNumber: {
    fontSize: 36,
    fontWeight: 'bold',
    color: 'white',
  },
  scoreMax: {
    fontSize: 18,
    color: 'white',
    opacity: 0.8,
  },
  scoreDescription: {
    textAlign: 'center',
    opacity: 0.8,
  },
  analysisSection: {
    marginBottom: 30,
  },
  itemsContainer: {
    gap: 10,
  },
  strengthItem: {
    backgroundColor: '#e8f5e8',
    padding: 15,
    borderRadius: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#4CAF50',
  },
  improvementItem: {
    backgroundColor: '#fff3e0',
    padding: 15,
    borderRadius: 8,
    borderLeftWidth: 4,
    borderLeftColor: '#FF9800',
  },
  itemText: {
    fontSize: 16,
  },
  drillsContainer: {
    gap: 15,
  },
  drillCard: {
    backgroundColor: '#f0f8ff',
    padding: 20,
    borderRadius: 12,
    borderWidth: 1,
    borderColor: '#e0e8ff',
  },
  drillTitle: {
    fontSize: 16,
    color: '#007AFF',
    marginBottom: 8,
  },
  drillText: {
    fontSize: 15,
    marginBottom: 12,
    lineHeight: 22,
  },
  drillMeta: {
    flexDirection: 'row',
    justifyContent: 'space-between',
  },
  drillDuration: {
    fontSize: 12,
    opacity: 0.7,
  },
  drillFrequency: {
    fontSize: 12,
    opacity: 0.7,
  },
  mechanicsSection: {
    marginBottom: 30,
  },
  mechanicsGrid: {
    flexDirection: 'row',
    flexWrap: 'wrap',
    gap: 10,
  },
  mechanicsCard: {
    backgroundColor: '#f9f9f9',
    padding: 15,
    borderRadius: 10,
    width: (width - 50) / 2,
    alignItems: 'center',
  },
  mechanicsTitle: {
    fontSize: 14,
    opacity: 0.8,
    marginBottom: 5,
  },
  mechanicsScore: {
    fontSize: 18,
    fontWeight: 'bold',
    color: '#007AFF',
    marginBottom: 5,
  },
  mechanicsNote: {
    fontSize: 12,
    opacity: 0.6,
    textAlign: 'center',
  },
  actionButtons: {
    gap: 15,
    marginBottom: 30,
  },
  primaryButton: {
    backgroundColor: '#007AFF',
    padding: 18,
    borderRadius: 12,
    alignItems: 'center',
  },
  primaryButtonText: {
    color: 'white',
    fontSize: 16,
    fontWeight: 'bold',
  },
  secondaryButton: {
    backgroundColor: 'transparent',
    borderWidth: 2,
    borderColor: '#007AFF',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
  },
  secondaryButtonText: {
    color: '#007AFF',
    fontSize: 16,
    fontWeight: 'bold',
  },
});
