// REQUIRED: Ensure supabase client is initialized at @/lib/supabase
import { ThemedText } from '@/components/ThemedText';
import { supabase } from '@/lib/supabase';
import { decode } from 'base64-arraybuffer';
import * as FileSystem from 'expo-file-system';
import * as ImagePicker from 'expo-image-picker';
import React, { useState } from 'react';
import {
  Alert,
  Dimensions,
  Pressable,
  ScrollView,
  StyleSheet,
  View,
  Image,
} from 'react-native';
import Animated, {
  Easing,
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  withDelay,
} from 'react-native-reanimated';
import { LinearGradient } from 'expo-linear-gradient';
import uuid from 'react-native-uuid';


const { width: SCREEN_WIDTH, height: SCREEN_HEIGHT } = Dimensions.get('window');

function formatTimeAgo(date: Date) {
  const now = new Date();
  const diff = Math.floor((now.getTime() - date.getTime()) / 1000);
  if (diff < 60) return 'Just now';
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} hours ago`;
  return `${Math.floor(diff / 86400)} days ago`;
}

type UploadType = 'swing' | 'pitch';

interface UploadHistoryItem {
  type: UploadType;
  score: number;
  date: Date;
}



export default function UploadScreen() {
  // UI-only state
  const [isUploading, setIsUploading] = useState(false);
  const [history, setHistory] = useState<UploadHistoryItem[]>([]);

  // ===== BUTTON GLOW + BUTTON SURGE (your original) =====
  const glowOpacity = useSharedValue(0.4);

  React.useEffect(() => {
    if (!isUploading) {
      glowOpacity.value = withRepeat(
        withTiming(1, { duration: 1500, easing: Easing.inOut(Easing.ease) }),
        -1,
        true
      );
    } else {
      glowOpacity.value = withTiming(0, { duration: 300 });
    }
  }, [isUploading]);

  const animatedGlow = useAnimatedStyle(() => ({
    shadowOpacity: glowOpacity.value,
  }));

  const animatedSurge = useAnimatedStyle(() => ({
    transform: [{ translateX: withTiming(isUploading ? 0 : -400, { duration: 0 }) }],
  }));

  // ===== CARD-SCOPED LIGHTNING (behind text, inside card) =====
  const cardPulse = useSharedValue(0);
  const cardWidth = SCREEN_WIDTH < 400 ? SCREEN_WIDTH : 380;

  // Left→Right / Right→Left / Diagonals / Vertical-ish (inside card)
  const cardStreakX = Array.from({ length: 8 }, () => useSharedValue(-360));
  const cardStreakY = Array.from({ length: 2 }, () => useSharedValue(-260));

  React.useEffect(() => {
    if (!isUploading) {
      // Pulse
      cardPulse.value = withRepeat(
        withTiming(1, { duration: 1700, easing: Easing.inOut(Easing.ease) }),
        -1,
        true
      );

      // Reset positions
      cardStreakX.forEach((v, i) => (cardStreakX[i].value = i < 2 ? 500 : -360));
      cardStreakY.forEach((v, i) => (cardStreakY[i].value = -260));

      // Left->Right
      cardStreakX[2].value = withDelay(
        0,
        withRepeat(withTiming(cardWidth + 320, { duration: 2600, easing: Easing.linear }), -1, false)
      );
      cardStreakX[3].value = withDelay(
        800,
        withRepeat(withTiming(cardWidth + 380, { duration: 3000, easing: Easing.linear }), -1, false)
      );
      cardStreakX[4].value = withDelay(
        1600,
        withRepeat(withTiming(cardWidth + 300, { duration: 2400, easing: Easing.linear }), -1, false)
      );

      // Right->Left
      cardStreakX[0].value = withDelay(
        600,
        withRepeat(withTiming(-420, { duration: 3200, easing: Easing.linear }), -1, false)
      );
      cardStreakX[1].value = withDelay(
        1400,
        withRepeat(withTiming(-470, { duration: 2800, easing: Easing.linear }), -1, false)
      );

      // Extra diagonals (also L->R but different top positions)
      cardStreakX[5].value = withDelay(
        1000,
        withRepeat(withTiming(cardWidth + 360, { duration: 3600, easing: Easing.linear }), -1, false)
      );
      cardStreakX[6].value = withDelay(
        2000,
        withRepeat(withTiming(cardWidth + 420, { duration: 4200, easing: Easing.linear }), -1, false)
      );
      cardStreakX[7].value = withDelay(
        2600,
        withRepeat(withTiming(cardWidth + 340, { duration: 3800, easing: Easing.linear }), -1, false)
      );

      // Vertical-ish (top->bottom)
      cardStreakY[0].value = withDelay(
        1200,
        withRepeat(withTiming(440, { duration: 3600, easing: Easing.linear }), -1, false)
      );
      cardStreakY[1].value = withDelay(
        1900,
        withRepeat(withTiming(480, { duration: 4200, easing: Easing.linear }), -1, false)
      );
    } else {
      // Fade FX when uploading
      cardPulse.value = withTiming(0, { duration: 250 });
      cardStreakX.forEach((_, i) => {
        cardStreakX[i].value = withTiming(i < 2 ? 500 : -360, { duration: 250 });
      });
      cardStreakY.forEach((_, i) => {
        cardStreakY[i].value = withTiming(-260, { duration: 250 });
      });
    }
  }, [isUploading]);

  const cardPulseStyle = useAnimatedStyle(() => ({
    opacity: 0.06 + cardPulse.value * 0.14,
  }));

  // angle helpers for the 8 X-based streaks
  const cardAngles = ['-18deg', '14deg', '22deg', '-12deg', '32deg', '-28deg', '45deg', '-35deg'];

  const cardStreakStyles = cardStreakX.map((sv, i) =>
    useAnimatedStyle(() => ({
      transform: [{ translateX: sv.value }, { rotateZ: cardAngles[i] as any }],
    }))
  );

  const cardStreakVStyles = cardStreakY.map((sv, i) =>
    useAnimatedStyle(() => ({
      transform: [{ translateY: sv.value }, { rotateZ: i === 0 ? '78deg' : '-82deg' }],
    }))
  );

  // ===== BACKGROUND (screen-wide) LIGHTNING for extra atmosphere =====
  const bgPulse = useSharedValue(0);
  const bgX = Array.from({ length: 6 }, () => useSharedValue(-340));
  const bgY = Array.from({ length: 2 }, () => useSharedValue(-220));

  React.useEffect(() => {
    if (!isUploading) {
      bgPulse.value = withRepeat(
        withTiming(1, { duration: 2200, easing: Easing.inOut(Easing.ease) }),
        -1,
        true
      );

      // Reset
      bgX[0].value = -300;
      bgX[1].value = -350;
      bgX[2].value = -280;
      bgX[3].value = -400;
      bgX[4].value = SCREEN_WIDTH + 220; // reverse
      bgX[5].value = SCREEN_WIDTH + 270; // reverse
      bgY[0].value = -180;
      bgY[1].value = -220;

      // L->R
      bgX[0].value = withDelay(
        200,
        withRepeat(withTiming(SCREEN_WIDTH + 320, { duration: 4200, easing: Easing.linear }), -1, false)
      );
      bgX[1].value = withDelay(
        1000,
        withRepeat(withTiming(SCREEN_WIDTH + 360, { duration: 5200, easing: Easing.linear }), -1, false)
      );
      bgX[2].value = withDelay(
        1800,
        withRepeat(withTiming(SCREEN_WIDTH + 300, { duration: 3600, easing: Easing.linear }), -1, false)
      );
      bgX[3].value = withDelay(
        2600,
        withRepeat(withTiming(SCREEN_WIDTH + 420, { duration: 5800, easing: Easing.linear }), -1, false)
      );

      // R->L
      bgX[4].value = withDelay(
        800,
        withRepeat(withTiming(-420, { duration: 4800, easing: Easing.linear }), -1, false)
      );
      bgX[5].value = withDelay(
        1600,
        withRepeat(withTiming(-470, { duration: 4200, easing: Easing.linear }), -1, false)
      );

      // Vertical-ish
      bgY[0].value = withDelay(
        1200,
        withRepeat(withTiming(SCREEN_HEIGHT + 180, { duration: 4600, easing: Easing.linear }), -1, false)
      );
      bgY[1].value = withDelay(
        2000,
        withRepeat(withTiming(SCREEN_HEIGHT + 240, { duration: 5200, easing: Easing.linear }), -1, false)
      );
    } else {
      bgPulse.value = withTiming(0, { duration: 250 });
      bgX.forEach((_, i) => {
        bgX[i].value = withTiming(i < 4 ? -340 : SCREEN_WIDTH + 240, { duration: 250 });
      });
      bgY.forEach((_, i) => {
        bgY[i].value = withTiming(-220, { duration: 250 });
      });
    }
  }, [isUploading]);

  const bgPulseStyle = useAnimatedStyle(() => ({
    opacity: 0.06 + bgPulse.value * 0.12,
  }));

  const bgAngles = ['22deg', '-18deg', '35deg', '-28deg', '25deg', '-32deg'];
  const bgStreakStyles = bgX.map((sv, i) =>
    useAnimatedStyle(() => ({
      transform: [{ translateX: sv.value }, { rotateZ: bgAngles[i] as any }],
    }))
  );
  const bgStreakVStyles = bgY.map((sv, i) =>
    useAnimatedStyle(() => ({
      transform: [{ translateY: sv.value }, { rotateZ: i === 0 ? '80deg' : '-78deg' }],
    }))
  );

  // ====== CORE LOGIC (unchanged) ======
  async function uriToBlob(uri: string): Promise<Blob> {
    const res = await fetch(uri);
    const blob = await res.blob();
    return blob;
  }
  

  async function uploadVideoToSupabase(fileUri: string, type: UploadType): Promise<{ signedUrl: string; filePath: string }> {
    try {
      const base64 = await FileSystem.readAsStringAsync(fileUri, {
        encoding: FileSystem.EncodingType.Base64,
      });

      const { data: sessionData } = await supabase.auth.getSession();
      if (!sessionData.session || !sessionData.session.user) {
        throw new Error('User is not authenticated');
      }

      const uniqueId = uuid.v4() as string;
      const filePath = `${type}-${uniqueId}.mp4`;
      const contentType = 'video/mp4';

      const { error } = await supabase.storage
        .from('mechanics-uploads-bucket')
        .upload(filePath, decode(base64), {
          contentType,
          upsert: true,
        });

      if (error) {
        throw new Error(`Upload failed: ${error.message}`);
      }

      const { data: signedUrlData, error: signedUrlError } = await supabase.storage
        .from('mechanics-uploads-bucket')
        .createSignedUrl(filePath, 60);

      if (signedUrlError || !signedUrlData?.signedUrl) {
        throw new Error('Failed to generate signed URL');
      }

      return { signedUrl: signedUrlData.signedUrl, filePath };
    } catch (err: any) {
      throw err;
    }
  }

  async function deleteVideoFromSupabase(filePath: string): Promise<boolean> {
    try {
      const { error } = await supabase.storage
        .from('mechanics-uploads-bucket')
        .remove([filePath]);

      if (error) {
        console.error('Failed to delete video:', error.message);
        return false;
      }

      console.log('✅ Deleted video:', filePath);
      return true;
    } catch (err: any) {
      console.error('Error deleting video from Supabase:', err.message);
      return false;
    }
  }


  async function analyzeWithBackend(publicUrl: string): Promise<any[]> {
    try {
      const response = await fetch(
        `http://192.168.1.19:8000/analyze?public_url=${encodeURIComponent(publicUrl)}`
      );
      const result = await response.json();
      Alert.alert('Analysis Result', result.feedback.replace(/\\n/g, '\n'));
      return result.feedback;
    } catch (err) {
      Alert.alert('Error', 'Analysis failed. Please try again.');
      return [];
    }
  }

  async function handleVideoUpload(type: UploadType, fileUri: string) {
    setIsUploading(true);

    let filePath: string | null = null;

    try {
      const { signedUrl, filePath: path } = await uploadVideoToSupabase(fileUri, type);
      filePath = path;

      let retries = 0;
      let success = false;
      let lastStatus = 0;

      while (retries < 10 && !success) {
        const response = await fetch(signedUrl);
        lastStatus = response.status;
        if (response.ok) {
          success = true;
          break;
        }
        await new Promise(resolve => setTimeout(resolve, 2000));
        retries++;
      }

      if (!success) {
        Alert.alert(
          'Upload not ready',
          `Video not available yet (last status: ${lastStatus}). Try again in a few seconds.`
        );
        return;
      }

      await new Promise(resolve => setTimeout(resolve, 5000));

      const feedback = await analyzeWithBackend(signedUrl);

      const score = +(7 + Math.random() * 3).toFixed(1);
      setHistory([{ type, score, date: new Date() }, ...history]);

    } catch (err: any) {
      Alert.alert('Upload Failed', err.message);
    } finally {
      setIsUploading(false);

      // ✅ CLEANUP: delete video from Supabase
      if (filePath) {
        //console.log('Cleaning up uploaded video from Supabase...');
        await deleteVideoFromSupabase(filePath);
      }
    }
  }


  const pickOrRecordVideo = async (source: 'camera' | 'library') => {
    let result;
    if (source === 'camera') {
      const permission = await ImagePicker.requestCameraPermissionsAsync();
      if (permission.status !== 'granted') {
        Alert.alert('Camera permission required', 'You must allow camera access to record a video.');
        return;
      }
      result = await ImagePicker.launchCameraAsync({ mediaTypes: 'videos', quality: 1 });
    } else {
      const permission = await ImagePicker.requestMediaLibraryPermissionsAsync();
      if (permission.status !== 'granted') {
        Alert.alert('Library permission required', 'You must allow access to your media library.');
        return;
      }
      result = await ImagePicker.launchImageLibraryAsync({ mediaTypes: 'videos' });
    }

    if (!result.canceled && result.assets && result.assets.length > 0) {
      await handleVideoUpload('swing', result.assets[0].uri);
    }
  };

  const openCamera = () => {
    Alert.alert('Capture', 'Choose how to add your swing', [
      { text: 'Record Video', onPress: () => pickOrRecordVideo('camera') },
      { text: 'Choose from Library', onPress: () => pickOrRecordVideo('library') },
      { text: 'Cancel', style: 'cancel' },
    ]);
  };

  return (
    <ScrollView
      style={styles.bg}
      contentContainerStyle={styles.centerContent}
      keyboardShouldPersistTaps="handled"
    >
      {/* ===== BACKGROUND FX (screen-wide, behind everything) ===== */}
      <View pointerEvents="none" style={styles.bgEffects}>
        <Animated.View style={[StyleSheet.absoluteFillObject, bgPulseStyle]}>
          <LinearGradient
            colors={['#0f1f18', '#123224', '#0f1f18']}
            start={{ x: 0, y: 0 }}
            end={{ x: 1, y: 1 }}
            style={StyleSheet.absoluteFillObject}
          />
        </Animated.View>

        {/* BG diagonal streaks */}
        <Animated.View style={[styles.bgStreak, { top: 60 }, bgStreakStyles[0]]}>
          <LinearGradient colors={['transparent', '#ffffff65', '#ffffff45', 'transparent']}
            start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }} style={styles.bgStreakFill} />
        </Animated.View>
        <Animated.View style={[styles.bgStreak, { top: SCREEN_HEIGHT * 0.25 }, bgStreakStyles[1]]}>
          <LinearGradient colors={['transparent', '#ffffff55', '#ffffff35', 'transparent']}
            start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }} style={styles.bgStreakFill} />
        </Animated.View>
        <Animated.View style={[styles.bgStreak, { top: SCREEN_HEIGHT * 0.45 }, bgStreakStyles[2]]}>
          <LinearGradient colors={['transparent', '#ffffff70', '#ffffff50', 'transparent']}
            start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }} style={styles.bgStreakFill} />
        </Animated.View>
        <Animated.View style={[styles.bgStreak, { top: SCREEN_HEIGHT * 0.65 }, bgStreakStyles[3]]}>
          <LinearGradient colors={['transparent', '#ffffff45', '#ffffff25', 'transparent']}
            start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }} style={styles.bgStreakFill} />
        </Animated.View>

        {/* BG reverse direction */}
        <Animated.View style={[styles.bgStreak, { top: SCREEN_HEIGHT * 0.15 }, bgStreakStyles[4]]}>
          <LinearGradient colors={['transparent', '#ffffff60', '#ffffff40', 'transparent']}
            start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }} style={styles.bgStreakFill} />
        </Animated.View>
        <Animated.View style={[styles.bgStreak, { top: SCREEN_HEIGHT * 0.75 }, bgStreakStyles[5]]}>
          <LinearGradient colors={['transparent', '#ffffff50', '#ffffff30', 'transparent']}
            start={{ x: 0, y: 0 }} end={{ x: 1, y: 0 }} style={styles.bgStreakFill} />
        </Animated.View>

        {/* BG vertical-ish */}
        <Animated.View style={[styles.bgStreakVertical, { left: SCREEN_WIDTH * 0.2 }, bgStreakVStyles[0]]}>
          <LinearGradient colors={['transparent', '#ffffff55', '#ffffff35', 'transparent']}
            start={{ x: 0, y: 0 }} end={{ x: 0, y: 1 }} style={styles.bgStreakFill} />
        </Animated.View>
        <Animated.View style={[styles.bgStreakVertical, { left: SCREEN_WIDTH * 0.7 }, bgStreakVStyles[1]]}>
          <LinearGradient colors={['transparent', '#ffffff45', '#ffffff25', 'transparent']}
            start={{ x: 0, y: 0 }} end={{ x: 0, y: 1 }} style={styles.bgStreakFill} />
        </Animated.View>
      </View>

      {/* ===== FOREGROUND CARD ===== */}
      <View style={styles.card}>
        {/* Card FX: inside card, behind all content */}
        <View pointerEvents="none" style={styles.cardFxLayer}>
          <Animated.View style={[StyleSheet.absoluteFillObject, cardPulseStyle]}>
            <LinearGradient
              colors={['#11271e', '#173427', '#11271e']}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={StyleSheet.absoluteFillObject}
            />
          </Animated.View>

          {/* 8 diagonal/angled streaks at various rows */}
          {cardStreakStyles.map((st, i) => (
            <Animated.View
              key={`cardx-${i}`}
              style={[styles.cardStreak, { top: 40 + i * 36 }, st]}
            >
              <LinearGradient
                colors={['transparent', i % 2 === 0 ? '#ffffff80' : '#ffffff60', 'transparent']}
                start={{ x: 0, y: 0 }}
                end={{ x: 1, y: 0 }}
                style={styles.cardStreakFill}
              />
            </Animated.View>
          ))}

          {/* 2 vertical-ish streaks */}
          {cardStreakVStyles.map((st, i) => (
            <Animated.View
              key={`cardy-${i}`}
              style={[
                styles.cardStreakVertical,
                { left: 40 + i * (cardWidth - 120) },
                st,
              ]}
            >
              <LinearGradient
                colors={['transparent', i === 0 ? '#ffffff70' : '#ffffff50', 'transparent']}
                start={{ x: 0, y: 0 }}
                end={{ x: 0, y: 1 }}
                style={styles.cardStreakFill}
              />
            </Animated.View>
          ))}
        </View>

        {/* Card content above FX */}
        <View style={styles.cardContent}>
          <View style={styles.header}>
            <ThemedText type="title" style={styles.headerTitle}>
              SwingGPT
            </ThemedText>
          </View>

          <View style={styles.section}>
            <View style={styles.futuristicPanel}>
              <ThemedText type="subtitle" style={[styles.sectionTitle, { textAlign: 'center' }]}>
                Setup For Best Results
              </ThemedText>

              <View style={styles.tipsContainer}>
                {/* Film swing from open side */}
                <View style={[styles.tip, { flexDirection: 'row', alignItems: 'center', marginBottom: 16 }]}>
                  <ThemedText style={[styles.tipEmoji, { fontSize: 22, marginRight: 8 }]}>📹</ThemedText>
                  <ThemedText style={[styles.tipText, { flex: 1 }]} numberOfLines={2}>
                    Film swing from open side ↙
                  </ThemedText>
                </View>

                <Image
                  source={require('../../assets/Untitled 233tt.gif')}
                  style={{ width: '100%', height: 120, resizeMode: 'contain', borderRadius: 8, marginBottom: 16 }}
                />

                {/* Other tips */}
                <View style={styles.tip}>
                  <ThemedText style={styles.tipEmoji}>🎯</ThemedText>
                  <ThemedText style={styles.tipText}>
                    Position 25-30 ft away, full body visible at all times
                  </ThemedText>
                </View>

                <View style={styles.tip}>
                  <ThemedText style={styles.tipEmoji}>💡</ThemedText>
                  <ThemedText style={styles.tipText}>
                    Good lighting, high quality & 30 fps, few/no shadows
                  </ThemedText>
                </View>

                <View style={styles.tip}>
                  <ThemedText style={styles.tipEmoji}>⏱️</ThemedText>
                  <ThemedText style={styles.tipText}>
                    Just capture the swing — no extra footage before or after
                  </ThemedText>
                </View>

                <View style={styles.tip}>
                  <ThemedText style={styles.tipEmoji}>👥</ThemedText>
                  <ThemedText style={styles.tipText}>
                    No one else in the frame
                  </ThemedText>
                </View>
              </View>

              <Animated.View style={[styles.glowWrapper, animatedGlow]}>
                <Pressable
                  style={[styles.uploadButton, isUploading && styles.uploadButtonDisabled]}
                  onPress={openCamera}
                  disabled={isUploading}
                >
                  <ThemedText style={styles.uploadButtonText}>
                    {isUploading ? 'ANALYZING...' : 'UPLOAD ↑'}
                  </ThemedText>

                  {isUploading && (
                    <Animated.View style={[styles.surgeBar, animatedSurge]}>
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
            </View>
          </View>
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  bg: {
    flex: 1,
    backgroundColor: '#082419',
  },
  centerContent: {
    alignItems: 'center',
    justifyContent: 'flex-start',
    paddingTop: 30,
    paddingBottom: 30,
    minHeight: '100%',
  },

  // Screen-wide background FX
  bgEffects: {
    ...StyleSheet.absoluteFillObject,
    zIndex: -1,
  },
  bgStreak: {
    position: 'absolute',
    left: -360,
    width: 320,
    height: 3,
    opacity: 0.33,
    borderRadius: 3,
  },
  bgStreakVertical: {
    position: 'absolute',
    top: -240,
    width: 3,
    height: 320,
    opacity: 0.28,
    borderRadius: 3,
  },
  bgStreakFill: {
    width: '100%',
    height: '100%',
    borderRadius: 3,
  },

  // Card
  card: {
    width: SCREEN_WIDTH < 400 ? '100%' : 380,
    backgroundColor: '#1a2e25',
    borderRadius: 20,
    padding: 0,
    marginHorizontal: 10,
    borderWidth: 2,
    borderColor: '#ffffff33',
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.2,
    shadowRadius: 20,
    elevation: 6,
    alignItems: 'stretch',
    position: 'relative',
    overflow: 'hidden', // clip streaks to card radius
  },
  cardFxLayer: {
    ...StyleSheet.absoluteFillObject,
    zIndex: 0,
  },
  cardContent: {
    padding: 24,
    zIndex: 1,
  },
  cardStreak: {
    position: 'absolute',
    left: -340,
    width: 320,
    height: 3,
    opacity: 0.42,
    borderRadius: 3,
  },
  cardStreakVertical: {
    position: 'absolute',
    top: -220,
    width: 3,
    height: 300,
    opacity: 0.40,
    borderRadius: 3,
  },
  cardStreakFill: {
    width: '100%',
    height: '100%',
    borderRadius: 3,
  },

  header: {
    alignItems: 'center',
    marginBottom: 20,
    gap: 8,
  },
  headerTitle: {
    fontSize: 26,
    fontWeight: '900',
    color: '#ffffff',
    letterSpacing: 2,
    textShadowColor: '#f44336',
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 10,
  },
  headerSubtitle: {
    color: '#d5ead6',
    fontSize: 15,
    textAlign: 'center',
    fontWeight: '500',
  },
  section: {
    marginBottom: 24,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: '800',
    color: '#f44336',
    marginBottom: 14,
    letterSpacing: 1,
    textTransform: 'uppercase',
  },
  futuristicPanel: {
    backgroundColor: '#0f1f18',
    padding: 20,
    borderRadius: 18,
    borderWidth: 1,
    borderColor: '#ffffff22',
    shadowColor: '#000',
    shadowOpacity: 0.15,
    shadowOffset: { width: 0, height: 6 },
    shadowRadius: 14,
  },
  tipsContainer: {
    backgroundColor: '#1f3a30',
    padding: 16,
    borderRadius: 12,
    marginBottom: 18,
    borderWidth: 1,
    borderColor: '#ffffff22',
  },
  tip: {
    flexDirection: 'row',
    alignItems: 'center',
    marginBottom: 12,
    gap: 8,
  },
  tipEmoji: {
    fontSize: 20,
    marginRight: 10,
  },
  tipText: {
    flex: 1,
    fontSize: 14,
    color: '#e3f9ec',
    fontWeight: '500',
  },
  uploadButton: {
    backgroundColor: '#f44336',
    paddingVertical: 18,
    borderRadius: 16,
    alignItems: 'center',
    marginBottom: 15,
    borderWidth: 2,
    borderColor: '#ffffff88',
    shadowColor: '#f44336',
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.3,
    shadowRadius: 12,
    elevation: 6,
    overflow: 'hidden',
  },
  uploadButtonDisabled: {
    opacity: 0.65,
  },
  uploadButtonText: {
    color: '#fff',
    fontSize: 18,
    fontWeight: '900',
    letterSpacing: 1.1,
    textTransform: 'uppercase',
  },
  processingContainer: {
    backgroundColor: 'rgba(255,255,255,0.06)',
    padding: 16,
    borderRadius: 12,
    alignItems: 'center',
    marginTop: 6,
    borderWidth: 1,
    borderColor: 'rgba(255,255,255,0.2)',
  },
  processingText: {
    fontSize: 16,
    marginBottom: 4,
    color: '#fff',
    fontWeight: '700',
    textTransform: 'uppercase',
    letterSpacing: 0.8,
  },
  processingSubtext: {
    opacity: 0.85,
    fontSize: 13,
    color: '#ddd',
    fontWeight: '500',
  },
  emptyState: {
    opacity: 0.7,
    textAlign: 'center',
    color: '#d9e7f5',
    fontSize: 15,
    fontStyle: 'italic',
  },
  historyItem: {
    flexDirection: 'row',
    alignItems: 'center',
    backgroundColor: '#20362e',
    borderRadius: 12,
    paddingVertical: 14,
    paddingHorizontal: 16,
    marginBottom: 10,
    borderWidth: 1,
    borderColor: '#ffffff15',
  },
  historyBadge: {
    width: 32,
    height: 32,
    borderRadius: 16,
    backgroundColor: 'rgba(255,255,255,0.08)',
    alignItems: 'center',
    justifyContent: 'center',
    marginRight: 14,
    borderWidth: 1,
    borderColor: '#ffffff33',
  },
  historyBadgeText: {
    fontWeight: '900',
    color: '#ffffff',
    fontSize: 13,
  },
  historyInfo: {
    flex: 1,
  },
  historyTitle: {
    fontSize: 16,
    fontWeight: '700',
    color: '#f5f9ff',
  },
  historyDate: {
    opacity: 0.8,
    fontSize: 13,
    marginTop: 2,
    color: '#a8cbb2',
    fontWeight: '500',
  },
  scoreContainer: {
    alignItems: 'center',
    backgroundColor: '#1a3f2b',
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 8,
    borderWidth: 1,
    borderColor: '#66bb6a',
  },
  historyScore: {
    fontSize: 18,
    fontWeight: '900',
    color: '#66bb6a',
  },
  scoreLabel: {
    fontSize: 12,
    fontWeight: '600',
    color: '#a5d6a7',
    marginTop: -2,
  },

  // Button glow/surge
  glowWrapper: {
    shadowColor: '#ffffff',
    shadowOffset: { width: 0, height: 0 },
    shadowRadius: 18,
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
});
