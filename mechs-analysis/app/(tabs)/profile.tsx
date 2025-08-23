import { useEffect, useState } from 'react';
import { router } from 'expo-router';
import { supabase } from '@/lib/supabase';
import {
  Dimensions,
  Pressable,
  ScrollView,
  StyleSheet,
  View,
  Alert,
} from 'react-native';
import { ThemedText } from '@/components/ThemedText';
import Animated, {
  Easing,
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
} from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';
import { Ionicons } from '@expo/vector-icons';

const { width: SCREEN_WIDTH } = Dimensions.get('window');

export default function ProfileScreen() {
  const [user, setUser] = useState<any>(null);
  const [userProfile, setUserProfile] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [signingOut, setSigningOut] = useState(false);

  const glow = useSharedValue(0.4);

  useEffect(() => {
    fetchUserData();
  }, []);

  useEffect(() => {
    if (!signingOut) {
      glow.value = withRepeat(
        withTiming(1, { duration: 1500, easing: Easing.inOut(Easing.ease) }),
        -1,
        true
      );
    } else {
      glow.value = withTiming(0, { duration: 300 });
    }
  }, [signingOut]);

  const glowStyle = useAnimatedStyle(() => ({
    shadowOpacity: glow.value,
  }));

  const fetchUserData = async () => {
    try {
      const { data: sessionData } = await supabase.auth.getSession();
      if (!sessionData.session) {
        router.replace('/');
        return;
      }

      const currentUser = sessionData.session.user;
      setUser(currentUser);

      // Fetch user profile data from your User table
      const { data: profileData, error } = await supabase
        .from('User')
        .select('username, full_name')
        .eq('email', currentUser.email)
        .maybeSingle();

      if (error) {
        console.error('Error fetching profile:', error);
      } else {
        setUserProfile(profileData);
      }
    } catch (error) {
      console.error('Error:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleSignOut = async () => {
    Alert.alert(
      'Sign Out',
      'Are you sure you want to sign out?',
      [
        {
          text: 'Cancel',
          style: 'cancel',
        },
        {
          text: 'Sign Out',
          style: 'destructive',
          onPress: async () => {
            setSigningOut(true);
            const { error } = await supabase.auth.signOut();
            if (error) {
              console.error('Error signing out:', error);
              Alert.alert('Error', 'Failed to sign out. Please try again.');
              setSigningOut(false);
            } else {
              router.replace('/');
            }
          },
        },
      ]
    );
  };

  if (loading) {
    return (
      <View style={styles.container}>
        <View style={styles.bg} pointerEvents="none">
          <LinearGradient
            colors={['#0f1f18', '#123224', '#0f1f18']}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={StyleSheet.absoluteFillObject}
          />
        </View>
        <View style={styles.loadingContainer}>
          <ThemedText style={styles.loadingText}>Loading profile...</ThemedText>
        </View>
      </View>
    );
  }

  return (
    <View style={styles.container}>
      <View style={styles.bg} pointerEvents="none">
        <LinearGradient
          colors={['#0f1f18', '#123224', '#0f1f18']}
          start={{ x: 0, y: 0 }}
          end={{ x: 1, y: 1 }}
          style={StyleSheet.absoluteFillObject}
        />
      </View>

      <ScrollView
        contentContainerStyle={styles.scrollContent}
        showsVerticalScrollIndicator={false}
      >
        <View style={styles.card}>
          <View style={styles.iconContainer}>
            <Ionicons name="person-circle" size={80} color="#f44336" />
          </View>
          
          <ThemedText type="title" style={styles.title}>Profile</ThemedText>
          
          <View style={styles.infoSection}>
            <View style={styles.infoRow}>
              <ThemedText style={styles.label}>Full Name</ThemedText>
              <ThemedText style={styles.value}>
                {userProfile?.full_name || 'Not provided'}
              </ThemedText>
            </View>

            <View style={styles.infoRow}>
              <ThemedText style={styles.label}>Username</ThemedText>
              <ThemedText style={styles.value}>
                {userProfile?.username || 'N/A'}
              </ThemedText>
            </View>

            <View style={styles.infoRow}>
              <ThemedText style={styles.label}>Email</ThemedText>
              <ThemedText style={styles.value}>
                {user?.email || 'N/A'}
              </ThemedText>
            </View>
          </View>

          <Animated.View style={[styles.glowWrapper, glowStyle]}>
            <Pressable
              style={[styles.signOutButton, signingOut && styles.disabledButton]}
              onPress={handleSignOut}
              disabled={signingOut}
            >
              <Ionicons 
                name="log-out-outline" 
                size={20} 
                color="#fff" 
                style={styles.buttonIcon}
              />
              <ThemedText style={styles.buttonText}>
                {signingOut ? 'Signing Out...' : 'Sign Out'}
              </ThemedText>
            </Pressable>
          </Animated.View>
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
  loadingContainer: {
    flex: 1,
    justifyContent: 'center',
    alignItems: 'center',
  },
  loadingText: {
    color: '#d5ead6',
    fontSize: 16,
    fontWeight: '500',
  },
  scrollContent: {
    paddingBottom: 40,
    paddingHorizontal: 24,
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
  iconContainer: {
    alignItems: 'center',
    marginBottom: 16,
  },
  title: {
    fontSize: 26,
    fontWeight: '900',
    color: '#ffffff',
    letterSpacing: 2,
    textAlign: 'center',
    marginBottom: 24,
    textShadowColor: '#f44336',
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 10,
  },
  infoSection: {
    marginBottom: 32,
  },
  infoRow: {
    marginBottom: 16,
    paddingBottom: 16,
    borderBottomWidth: 1,
    borderBottomColor: '#ffffff22',
  },
  label: {
    color: '#e0f2f1',
    fontWeight: '600',
    fontSize: 15,
    marginBottom: 6,
  },
  value: {
    color: '#d5ead6',
    fontSize: 16,
    fontWeight: '500',
  },
  signOutButton: {
    backgroundColor: '#f44336',
    paddingVertical: 16,
    borderRadius: 16,
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'row',
    borderWidth: 2,
    borderColor: '#ffffff88',
    shadowColor: '#f44336',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.3,
    shadowRadius: 12,
    elevation: 6,
  },
  disabledButton: {
    opacity: 0.65,
  },
  buttonIcon: {
    marginRight: 8,
  },
  buttonText: {
    color: '#fff',
    fontSize: 16,
    fontWeight: '900',
    letterSpacing: 1,
    textTransform: 'uppercase',
  },
  glowWrapper: {
    shadowColor: '#ffffff',
    shadowOffset: { width: 0, height: 0 },
    shadowRadius: 18,
  },
});