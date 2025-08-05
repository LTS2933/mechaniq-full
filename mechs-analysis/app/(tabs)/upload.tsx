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
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  View,
} from 'react-native';

const SCREEN_WIDTH = Dimensions.get('window').width;

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
  const [selectedType, setSelectedType] = useState<UploadType | null>(null);
  const [isUploading, setIsUploading] = useState(false);
  const [history, setHistory] = useState<UploadHistoryItem[]>([]);

  // Convert local file URI to blob for upload
  async function uriToBlob(uri: string): Promise<Blob> {
    const res = await fetch(uri);
    //console.log("Fetch status:", res.status);
    const blob = await res.blob();
    /* console.log("Uploading from URI:", uri);
    console.log("BLOB DEBUG =======");
    console.log("Type:", blob.type);
    console.log("Size (bytes):", blob.size); */
    return blob;
  }

  async function uploadVideoToSupabase(fileUri: string, type: UploadType): Promise<string> {
    try {
      const base64 = await FileSystem.readAsStringAsync(fileUri, {
        encoding: FileSystem.EncodingType.Base64,
      });

      const { data: sessionData, error: sessionError } = await supabase.auth.getSession();

      if (!sessionData.session || !sessionData.session.user) {
        throw new Error("User is not authenticated");
      }

      const timestamp = Date.now();
      const filePath = `${type}-${timestamp}.mp4`;
      const contentType = 'video/mp4';

      const { data, error } = await supabase.storage
        .from('mechanics-uploads-bucket')
        .upload(filePath, decode(base64), {
          contentType,
          upsert: true,
        });

      if (error) {
        throw new Error(`Upload failed: ${error.message}`);
      }

      // 🔐 Use signed URL instead of public URL
      const { data: signedUrlData, error: signedUrlError } = await supabase.storage
        .from('mechanics-uploads-bucket')
        .createSignedUrl(filePath, 60); // expires in 60 seconds

      if (signedUrlError || !signedUrlData?.signedUrl) {
        throw new Error("Failed to generate signed URL");
      }

      console.log("✅ Upload successful with signed URL:", signedUrlData.signedUrl);
      return signedUrlData.signedUrl;
    } catch (err: any) {
      console.error("❌ Upload failed:", err.message || err);
      throw err;
    }
  }


  async function analyzeWithBackend(publicUrl: string): Promise<any[]> {
    try {
      const response = await fetch(`http://192.168.1.19:8000/analyze?public_url=${encodeURIComponent(publicUrl)}`);
      const result = await response.json();
      console.log("AI Feedback:", result.feedback);

      Alert.alert("Analysis Result", JSON.stringify(result.feedback, null, 2));
      return result.feedback;
    } catch (err) {
      console.error("❌ Analysis failed:", err);
      Alert.alert("Error", "Analysis failed. Please try again.");
      return [];
    }
  }


  async function handleVideoUpload(type: UploadType, fileUri: string) {
    setIsUploading(true);
    try {
      const videoUrl = await uploadVideoToSupabase(fileUri, type);

      let retries = 0;
      let success = false;
      let lastStatus = 0;

      while (retries < 10 && !success) {
        const response = await fetch(videoUrl);
        lastStatus = response.status;

        if (response.ok) {
          success = true;
          break;
        }

        console.log(`⏳ Waiting for video to be publicly accessible... retry ${retries + 1} (status ${response.status})`);
        await new Promise(resolve => setTimeout(resolve, 2000)); // wait 2 seconds
        retries++;
      }

      if (!success) {
        Alert.alert("Upload not ready", `Video not available yet (last status: ${lastStatus}). Try again in a few seconds.`);
        return;
      }



      await new Promise(resolve => setTimeout(resolve, 5000));

      // 🔁 NEW: Call backend for pose analysis
      const feedback = await analyzeWithBackend(videoUrl);

      const score = +(7 + Math.random() * 3).toFixed(1); // You can replace this with real scoring later
      setHistory([{ type, score, date: new Date() }, ...history]);

      Alert.alert(
        'Analysis Complete!',
        `Your ${type} video has been analyzed.\n\nTop Feedback:\n${feedback[0]?.issue ?? 'No issues found!'}`,
        [{ text: 'OK' }]
      );
    } catch (err: any) {
      Alert.alert('Upload Failed', err.message);
    } finally {
      setIsUploading(false);
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
      await handleVideoUpload(selectedType!, result.assets[0].uri);
    }
  };

  const openCamera = () => {
    if (!selectedType) {
      Alert.alert('Please select a type first.');
      return;
    }
    Alert.alert('Camera Options', 'Choose how to capture your video', [
      { text: 'Take Video', onPress: () => pickOrRecordVideo('camera') },
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
      <View style={styles.card}>
        <View style={styles.header}>
          <ThemedText type="title" style={styles.headerTitle}>
            Upload Video
          </ThemedText>
          <ThemedText style={styles.headerSubtitle}>
            Record your swing or pitch for AI-powered analysis
          </ThemedText>
        </View>

        <View style={styles.section}>
          <ThemedText type="subtitle" style={styles.sectionTitle}>
            What would you like to analyze?
          </ThemedText>

          <View style={styles.typeButtons}>
            <Pressable
              style={[
                styles.typeButton,
                selectedType === 'swing' && styles.selectedTypeButton,
              ]}
              onPress={() => setSelectedType('swing')}
            >
              <ThemedText
                style={[
                  styles.typeButtonText,
                  selectedType === 'swing' && styles.selectedTypeButtonText,
                ]}
              >
                🏏 Batting Swing
              </ThemedText>
              <ThemedText style={styles.typeDescription}>
                Analyze your batting technique, timing, and form
              </ThemedText>
            </Pressable>

            <Pressable
              style={[
                styles.typeButton,
                selectedType === 'pitch' && styles.selectedTypeButton,
              ]}
              onPress={() => setSelectedType('pitch')}
            >
              <ThemedText
                style={[
                  styles.typeButtonText,
                  selectedType === 'pitch' && styles.selectedTypeButtonText,
                ]}
              >
                ⚾ Pitching
              </ThemedText>
              <ThemedText style={styles.typeDescription}>
                Analyze your pitching mechanics and delivery
              </ThemedText>
            </Pressable>
          </View>
        </View>

        {selectedType && (
          <View style={styles.section}>
            <ThemedText type="subtitle" style={styles.sectionTitle}>
              Recording Tips
            </ThemedText>
            <View style={styles.tipsContainer}>
              <View style={styles.tip}>
                <ThemedText style={styles.tipEmoji}>📱</ThemedText>
                <ThemedText style={styles.tipText}>
                  Hold phone horizontally for best results
                </ThemedText>
              </View>
              <View style={styles.tip}>
                <ThemedText style={styles.tipEmoji}>📏</ThemedText>
                <ThemedText style={styles.tipText}>
                  Record from 10-15 feet away to capture full motion
                </ThemedText>
              </View>
              <View style={styles.tip}>
                <ThemedText style={styles.tipEmoji}>☀️</ThemedText>
                <ThemedText style={styles.tipText}>
                  Ensure good lighting and clear background
                </ThemedText>
              </View>
              <View style={styles.tip}>
                <ThemedText style={styles.tipEmoji}>⏱️</ThemedText>
                <ThemedText style={styles.tipText}>
                  Keep video 3-10 seconds for best analysis
                </ThemedText>
              </View>
            </View>
            <Pressable
              style={[
                styles.uploadButton,
                isUploading && { backgroundColor: '#b3cdfa' },
              ]}
              onPress={openCamera}
              disabled={isUploading}
            >
              <ThemedText style={styles.uploadButtonText}>
                {isUploading
                  ? 'Analyzing...'
                  : `Record ${selectedType === 'swing' ? 'Swing' : 'Pitch'}`}
              </ThemedText>
            </Pressable>
            {isUploading && (
              <View style={styles.processingContainer}>
                <ThemedText style={styles.processingText}>
                  🤖 AI is analyzing your technique...
                </ThemedText>
                <ThemedText style={styles.processingSubtext}>
                  This may take a few seconds
                </ThemedText>
              </View>
            )}
          </View>
        )}

        <View style={styles.section}>
          <ThemedText type="subtitle" style={styles.sectionTitle}>
            Recent Uploads
          </ThemedText>
          {history.length === 0 && (
            <ThemedText style={{ opacity: 0.5, textAlign: 'center' }}>
              No uploads yet. Try analyzing a video!
            </ThemedText>
          )}
          {history.map((item, idx) => (
            <View style={styles.historyItem} key={idx}>
              <View style={styles.historyBadge}>
                <ThemedText style={styles.historyBadgeText}>
                  {history.length - idx}
                </ThemedText>
              </View>
              <View style={styles.historyInfo}>
                <ThemedText type="defaultSemiBold" style={styles.historyTitle}>
                  {item.type === 'swing'
                    ? 'Swing Analysis'
                    : 'Pitch Analysis'}
                </ThemedText>
                <ThemedText style={styles.historyDate}>
                  {formatTimeAgo(item.date)}
                </ThemedText>
              </View>
              <ThemedText style={styles.historyScore}>{item.score}/10</ThemedText>
            </View>
          ))}
        </View>
      </View>
    </ScrollView>
  );
}

const styles = StyleSheet.create({
  bg: {
    flex: 1,
    backgroundColor: "#f4f8fc",
  },
  centerContent: {
    alignItems: "center",
    justifyContent: "flex-start",
    paddingTop: 30,
    paddingBottom: 30,
    minHeight: "100%",
  },
  card: {
    width: SCREEN_WIDTH < 400 ? "100%" : 360,
    backgroundColor: "#fff",
    borderRadius: 18,
    padding: 22,
    marginHorizontal: 10,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.06,
    shadowRadius: 14,
    elevation: 6,
    alignItems: "stretch",
  },
  header: {
    alignItems: "center",
    marginBottom: 16,
    gap: 6,
  },
  headerTitle: {
    fontSize: 24,
    fontWeight: "700",
    color: "#1850B5",
    marginBottom: 2,
  },
  headerSubtitle: {
    color: "#466",
    opacity: 0.77,
    fontSize: 15,
    textAlign: "center",
    marginBottom: 2,
  },
  section: {
    marginBottom: 22,
  },
  sectionTitle: {
    fontSize: 16,
    fontWeight: "600",
    color: "#1850B5",
    marginBottom: 11,
  },
  typeButtons: {
    gap: 13,
  },
  typeButton: {
    backgroundColor: "#f7f9fc",
    paddingVertical: 19,
    paddingHorizontal: 15,
    borderRadius: 13,
    borderWidth: 1.7,
    borderColor: "#e1e9f6",
    marginBottom: 10,
    elevation: 2,
  },
  selectedTypeButton: {
    backgroundColor: "#e6f3ff",
    borderColor: "#007AFF",
  },
  typeButtonText: {
    fontSize: 18,
    fontWeight: "bold",
    marginBottom: 6,
    color: "#22334a",
  },
  selectedTypeButtonText: {
    color: "#007AFF",
  },
  typeDescription: {
    opacity: 0.73,
    fontSize: 14,
    color: "#334a66",
  },
  tipsContainer: {
    backgroundColor: "#f9f9f9",
    padding: 17,
    borderRadius: 10,
    marginBottom: 15,
  },
  tip: {
    flexDirection: "row",
    alignItems: "center",
    marginBottom: 10,
    gap: 2,
  },
  tipEmoji: {
    fontSize: 19,
    marginRight: 10,
  },
  tipText: {
    flex: 1,
    fontSize: 14,
    opacity: 0.8,
    color: "#1d2940",
  },
  uploadButton: {
    backgroundColor: "#007AFF",
    paddingVertical: 15,
    borderRadius: 12,
    alignItems: "center",
    marginBottom: 13,
    elevation: 2,
  },
  uploadButtonText: {
    color: "white",
    fontSize: 17,
    fontWeight: "bold",
  },
  processingContainer: {
    backgroundColor: "#e7f3fe",
    padding: 16,
    borderRadius: 10,
    alignItems: "center",
    marginTop: 4,
  },
  processingText: {
    fontSize: 16,
    marginBottom: 4,
    color: "#1d2940",
  },
  processingSubtext: {
    opacity: 0.7,
    fontSize: 13,
    color: "#224",
  },
  // History
  historyItem: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "#f6f7fa",
    borderRadius: 8,
    paddingVertical: 12,
    paddingHorizontal: 14,
    marginBottom: 9,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: Platform.OS === "ios" ? 0.05 : 0.11,
    shadowRadius: 6,
    elevation: 1,
  },
  historyBadge: {
    width: 28,
    height: 28,
    borderRadius: 14,
    backgroundColor: "#b2d4fa",
    alignItems: "center",
    justifyContent: "center",
    marginRight: 11,
  },
  historyBadgeText: {
    fontWeight: "700",
    color: "#2267D5",
    fontSize: 15,
  },
  historyInfo: {
    flex: 1,
  },
  historyTitle: {
    fontSize: 15,
    fontWeight: "600",
    color: "#22334a",
  },
  historyDate: {
    opacity: 0.6,
    fontSize: 12,
    marginTop: 2,
    color: "#444",
  },
  historyScore: {
    fontSize: 16,
    fontWeight: "bold",
    color: "#007AFF",
    marginLeft: 7,
  },
});
