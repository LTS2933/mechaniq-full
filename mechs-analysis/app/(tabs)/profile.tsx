
import { ThemedText } from '@/components/ThemedText';
import { ThemedView } from '@/components/ThemedView';
import { Image } from 'expo-image';
import React, { useState } from 'react';
import { Alert, ScrollView, StyleSheet, View } from 'react-native';

export default function ProfileScreen() {
  const [user] = useState({
    name: 'Alex Johnson',
    email: 'alex.johnson@email.com',
    position: 'Shortstop',
    experience: 'High School Varsity',
    joinDate: 'March 2024',
    totalAnalyses: 24,
    avgScore: 8.2,
    improvementRate: '+12%'
  });

  const handleEditProfile = () => {
    Alert.alert('Edit Profile', 'Profile editing would open here');
  };

  const handleSettings = () => {
    Alert.alert('Settings', 'Settings screen would open here');
  };

  const handleContactCoach = () => {
    Alert.alert('Contact Coach', 'Coach contact feature would open here');
  };

  return (
    <ScrollView style={styles.container}>
      <ThemedView style={styles.header}>
        <ThemedText type="title">Profile</ThemedText>
      </ThemedView>

      <ThemedView style={styles.profileCard}>
        <Image
          source={require('@/assets/images/icon.png')}
          style={styles.avatar}
        />
        <ThemedText type="title" style={styles.userName}>{user.name}</ThemedText>
        <ThemedText style={styles.userEmail}>{user.email}</ThemedText>
        <ThemedText style={styles.userPosition}>{user.position} • {user.experience}</ThemedText>
        
        <View style={styles.editButton}>
          <ThemedText
            style={styles.editButtonText}
            onPress={handleEditProfile}
          >
            Edit Profile
          </ThemedText>
        </View>
      </ThemedView>

      <ThemedView style={styles.statsSection}>
        <ThemedText type="subtitle" style={styles.sectionTitle}>Your Stats</ThemedText>
        
        <View style={styles.statsGrid}>
          <View style={styles.statItem}>
            <ThemedText style={styles.statNumber}>{user.totalAnalyses}</ThemedText>
            <ThemedText style={styles.statLabel}>Total Analyses</ThemedText>
          </View>
          
          <View style={styles.statItem}>
            <ThemedText style={styles.statNumber}>{user.avgScore}</ThemedText>
            <ThemedText style={styles.statLabel}>Average Score</ThemedText>
          </View>
          
          <View style={styles.statItem}>
            <ThemedText style={[styles.statNumber, { color: '#4CAF50' }]}>
              {user.improvementRate}
            </ThemedText>
            <ThemedText style={styles.statLabel}>Improvement Rate</ThemedText>
          </View>
        </View>
      </ThemedView>

      <ThemedView style={styles.achievementsSection}>
        <ThemedText type="subtitle" style={styles.sectionTitle}>Achievements</ThemedText>
        
        <View style={styles.achievementsList}>
          <View style={styles.achievement}>
            <ThemedText style={styles.achievementIcon}>🏆</ThemedText>
            <View style={styles.achievementInfo}>
              <ThemedText type="defaultSemiBold">First Analysis</ThemedText>
              <ThemedText style={styles.achievementDate}>Completed March 15, 2024</ThemedText>
            </View>
          </View>
          
          <View style={styles.achievement}>
            <ThemedText style={styles.achievementIcon}>🔥</ThemedText>
            <View style={styles.achievementInfo}>
              <ThemedText type="defaultSemiBold">7-Day Streak</ThemedText>
              <ThemedText style={styles.achievementDate}>Consistent practice week</ThemedText>
            </View>
          </View>
          
          <View style={styles.achievement}>
            <ThemedText style={styles.achievementIcon}>📈</ThemedText>
            <View style={styles.achievementInfo}>
              <ThemedText type="defaultSemiBold">Score Improvement</ThemedText>
              <ThemedText style={styles.achievementDate}>+2.5 points this month</ThemedText>
            </View>
          </View>
          
          <View style={styles.achievement}>
            <ThemedText style={styles.achievementIcon}>🎯</ThemedText>
            <View style={styles.achievementInfo}>
              <ThemedText type="defaultSemiBold">Perfect Form</ThemedText>
              <ThemedText style={styles.achievementDate}>Scored 9.5+ on swing analysis</ThemedText>
            </View>
          </View>
        </View>
      </ThemedView>

      <ThemedView style={styles.progressSection}>
        <ThemedText type="subtitle" style={styles.sectionTitle}>Monthly Progress</ThemedText>
        
        <View style={styles.progressChart}>
          <View style={styles.monthItem}>
            <ThemedText style={styles.monthName}>Mar</ThemedText>
            <View style={styles.progressBar}>
              <View style={[styles.progressFill, { width: '70%' }]} />
            </View>
            <ThemedText style={styles.monthScore}>7.8</ThemedText>
          </View>
          
          <View style={styles.monthItem}>
            <ThemedText style={styles.monthName}>Apr</ThemedText>
            <View style={styles.progressBar}>
              <View style={[styles.progressFill, { width: '82%' }]} />
            </View>
            <ThemedText style={styles.monthScore}>8.2</ThemedText>
          </View>
          
          <View style={styles.monthItem}>
            <ThemedText style={styles.monthName}>May</ThemedText>
            <View style={styles.progressBar}>
              <View style={[styles.progressFill, { width: '85%' }]} />
            </View>
            <ThemedText style={styles.monthScore}>8.5</ThemedText>
          </View>
        </View>
      </ThemedView>

      <ThemedView style={styles.actionsSection}>
        <ThemedText type="subtitle" style={styles.sectionTitle}>Quick Actions</ThemedText>
        
        <View style={styles.actionsList}>
          <View style={styles.actionItem}>
            <ThemedText
              style={styles.actionText}
              onPress={handleContactCoach}
            >
              👨‍🏫 Contact Coach
            </ThemedText>
          </View>
          
          <View style={styles.actionItem}>
            <ThemedText
              style={styles.actionText}
              onPress={() => Alert.alert('Export', 'Export data feature')}
            >
              📊 Export Analysis Data
            </ThemedText>
          </View>
          
          <View style={styles.actionItem}>
            <ThemedText
              style={styles.actionText}
              onPress={() => Alert.alert('Support', 'Help & support feature')}
            >
              🆘 Help & Support
            </ThemedText>
          </View>
          
          <View style={styles.actionItem}>
            <ThemedText
              style={styles.actionText}
              onPress={handleSettings}
            >
              ⚙️ Settings
            </ThemedText>
          </View>
        </View>
      </ThemedView>

      <ThemedView style={styles.signOutSection}>
        <View style={styles.signOutButton}>
          <ThemedText style={styles.signOutText}>
            Sign Out
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
  profileCard: {
    backgroundColor: '#f9f9f9',
    padding: 30,
    borderRadius: 15,
    alignItems: 'center',
    marginBottom: 30,
  },
  avatar: {
    width: 80,
    height: 80,
    borderRadius: 40,
    marginBottom: 15,
  },
  userName: {
    marginBottom: 5,
  },
  userEmail: {
    opacity: 0.7,
    marginBottom: 5,
  },
  userPosition: {
    opacity: 0.6,
    fontSize: 14,
    marginBottom: 20,
  },
  editButton: {
    backgroundColor: '#007AFF',
    paddingHorizontal: 30,
    paddingVertical: 12,
    borderRadius: 25,
  },
  editButtonText: {
    color: 'white',
    fontWeight: 'bold',
  },
  statsSection: {
    marginBottom: 30,
  },
  sectionTitle: {
    marginBottom: 15,
  },
  statsGrid: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    gap: 10,
  },
  statItem: {
    backgroundColor: '#f9f9f9',
    padding: 20,
    borderRadius: 12,
    flex: 1,
    alignItems: 'center',
  },
  statNumber: {
    fontSize: 24,
    fontWeight: 'bold',
    color: '#007AFF',
    marginBottom: 5,
  },
  statLabel: {
    fontSize: 12,
    opacity: 0.7,
    textAlign: 'center',
  },
  achievementsSection: {
    marginBottom: 30,
  },
  achievementsList: {
    gap: 15,
  },
  achievement: {
    flexDirection: 'row',
    backgroundColor: '#f9f9f9',
    padding: 15,
    borderRadius: 12,
    alignItems: 'center',
  },
  achievementIcon: {
    fontSize: 30,
    marginRight: 15,
  },
  achievementInfo: {
    flex: 1,
  },
  achievementDate: {
    opacity: 0.6,
    fontSize: 12,
    marginTop: 4,
  },
  progressSection: {
    marginBottom: 30,
  },
  progressChart: {
    backgroundColor: '#f9f9f9',
    padding: 20,
    borderRadius: 12,
  },
  monthItem: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 15,
  },
  monthName: {
    width: 40,
    fontSize: 14,
    fontWeight: 'bold',
  },
  progressBar: {
    flex: 1,
    height: 8,
    backgroundColor: '#e0e0e0',
    borderRadius: 4,
    marginHorizontal: 15,
  },
  progressFill: {
    height: '100%',
    backgroundColor: '#007AFF',
    borderRadius: 4,
  },
  monthScore: {
    width: 40,
    textAlign: 'right',
    fontWeight: 'bold',
    color: '#007AFF',
  },
  actionsSection: {
    marginBottom: 30,
  },
  actionsList: {
    gap: 12,
  },
  actionItem: {
    backgroundColor: '#f9f9f9',
    padding: 18,
    borderRadius: 12,
  },
  actionText: {
    fontSize: 16,
    color: '#333',
  },
  signOutSection: {
    marginBottom: 30,
    alignItems: 'center',
  },
  signOutButton: {
    backgroundColor: '#ff4444',
    paddingHorizontal: 40,
    paddingVertical: 15,
    borderRadius: 25,
  },
  signOutText: {
    color: 'white',
    fontWeight: 'bold',
    fontSize: 16,
  },
});
