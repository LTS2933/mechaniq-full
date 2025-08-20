import { useEffect, useState } from 'react';
import { Redirect, router } from 'expo-router';
import { supabase } from '@/lib/supabase';
import {
  Dimensions,
  Keyboard,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  TextInput,
  TouchableWithoutFeedback,
  View,
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

const { width: SCREEN_WIDTH } = Dimensions.get('window');

export default function IndexScreen() {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [signedIn, setSignedIn] = useState(false);

  const glow = useSharedValue(0.4);
  const surge = useSharedValue(-400);

  useEffect(() => {
    const checkSession = async () => {
      const { data } = await supabase.auth.getSession();
      if (data.session) {
        setSignedIn(true);
      }
    };
    checkSession();
  }, []);

  useEffect(() => {
    if (!loading) {
      glow.value = withRepeat(
        withTiming(1, { duration: 1500, easing: Easing.inOut(Easing.ease) }),
        -1,
        true
      );
    } else {
      glow.value = withTiming(0, { duration: 300 });
      surge.value = withTiming(0, { duration: 0 });
    }
  }, [loading]);

  const glowStyle = useAnimatedStyle(() => ({
    shadowOpacity: glow.value,
  }));

  const surgeStyle = useAnimatedStyle(() => ({
    transform: [{ translateX: surge.value }],
  }));

  const handleSignIn = async () => {
    setLoading(true);
    setError(null);

    const { data: userRecord, error: userCheckError } = await supabase
      .from('User')
      .select('email')
      .eq('username', username)
      .maybeSingle();

    if (userCheckError || !userRecord) {
      setLoading(false);
      setError(userCheckError ? 'Error checking user.' : 'User not found.');
      return;
    }

    const { error: signInError } = await supabase.auth.signInWithPassword({
      email: userRecord.email,
      password,
    });

    if (signInError) {
      setError('Incorrect password.');
    } else {
      setSignedIn(true);
    }
    setLoading(false);
  };

  if (signedIn) return <Redirect href="/(tabs)/upload" />;

  return (
    <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === 'ios' ? 'padding' : 'height'}
        keyboardVerticalOffset={Platform.OS === 'ios' ? 0 : 20}
      >
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
            keyboardShouldPersistTaps="handled"
            showsVerticalScrollIndicator={false}
          >
            <View style={styles.card}>
              <ThemedText type="title" style={styles.title}>SwingGPT</ThemedText>
              <ThemedText style={styles.subtitle}>Log in to enter the swing lab</ThemedText>

              <View style={styles.inputGroup}>
                <ThemedText style={styles.label}>Username</ThemedText>
                <TextInput
                  style={styles.input}
                  placeholder="Username"
                  value={username}
                  onChangeText={setUsername}
                  autoCapitalize="none"
                />
              </View>

              <View style={styles.inputGroup}>
                <ThemedText style={styles.label}>Password</ThemedText>
                <TextInput
                  style={styles.input}
                  placeholder="Password"
                  value={password}
                  onChangeText={setPassword}
                  secureTextEntry
                  autoCapitalize="none"
                />
              </View>

              {error && (
                <ThemedText style={styles.error}>{error}</ThemedText>
              )}

              <Animated.View style={[styles.glowWrapper, glowStyle]}>
                <Pressable
                  style={[styles.button, loading && styles.disabledButton]}
                  onPress={handleSignIn}
                  disabled={loading}
                >
                  <ThemedText style={styles.buttonText}>
                    {loading ? 'Authenticating...' : 'Enter the Lab'}
                  </ThemedText>
                  {loading && (
                    <Animated.View style={[styles.surgeBar, surgeStyle]}>
                      <LinearGradient
                        colors={['transparent', '#ffffff66', 'transparent']}
                        start={{ x: 0, y: 0 }}
                        end={{ x: 1, y: 0 }}
                        style={styles.surgeLine}
                      />
                    </Animated.View>
                  )}
                </Pressable>
              </Animated.View>

              <Pressable
                style={({ pressed }) => [
                  styles.registerButton,
                  pressed && { opacity: 0.75 },
                ]}
                onPress={() => router.push('/(tabs)/register')}
              >
                <ThemedText style={styles.registerText}>
                  Don’t have an account? Register
                </ThemedText>
              </Pressable>
            </View>
          </ScrollView>
        </View>
      </KeyboardAvoidingView>
    </TouchableWithoutFeedback>
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
  },
  title: {
    fontSize: 26,
    fontWeight: '900',
    color: '#ffffff',
    letterSpacing: 2,
    textAlign: 'center',
    marginBottom: 6,
    textShadowColor: '#f44336',
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 10,
  },
  subtitle: {
    color: '#d5ead6',
    fontSize: 15,
    textAlign: 'center',
    fontWeight: '500',
    marginBottom: 20,
  },
  inputGroup: {
    marginBottom: 14,
  },
  label: {
    color: '#e0f2f1',
    fontWeight: '600',
    fontSize: 15,
    marginBottom: 6,
  },
  input: {
    height: 44,
    backgroundColor: '#243c32',
    borderRadius: 10,
    paddingHorizontal: 12,
    color: '#fff',
    borderWidth: 1,
    borderColor: '#ffffff22',
    fontSize: 15,
  },
  error: {
    color: '#f88',
    fontSize: 14,
    fontWeight: '600',
    textAlign: 'center',
    marginTop: 6,
  },
  button: {
    backgroundColor: '#f44336',
    paddingVertical: 16,
    borderRadius: 16,
    alignItems: 'center',
    marginTop: 16,
    borderWidth: 2,
    borderColor: '#ffffff88',
    shadowColor: '#f44336',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.3,
    shadowRadius: 12,
    elevation: 6,
    overflow: 'hidden',
  },
  disabledButton: {
    opacity: 0.65,
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
    marginTop: 12,
  },
  surgeBar: {
    position: 'absolute',
    left: -400,
    top: 0,
    bottom: 0,
    width: 400,
    zIndex: 2,
  },
  surgeLine: {
    flex: 1,
    opacity: 0.3,
  },
  registerButton: {
    marginTop: 14,
    alignItems: 'center',
  },
  registerText: {
    color: '#a5d6a7',
    fontWeight: '600',
    fontSize: 14,
    textDecorationLine: 'underline',
  },
});
