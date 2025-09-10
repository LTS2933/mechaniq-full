'use client';

import React, { useState, useCallback, useEffect } from 'react';
import {
  View,
  StyleSheet,
  FlatList,
  TouchableOpacity,
  Text,
  ActivityIndicator,
  Alert,
  Image,
} from 'react-native';
import { ThemedText } from '@/components/ThemedText';
import { supabase } from '@/lib/supabase';
import { Ionicons } from '@expo/vector-icons';
import * as FileSystem from 'expo-file-system';
import * as Sharing from 'expo-sharing';
import { useFocusEffect } from '@react-navigation/native';
import { Video, ResizeMode } from 'expo-av';
import Modal from 'react-native-modal';

interface Swing {
  id: string;
  user_id: string;
  s3_key: string;
  uniqueid: string;
  feedback: string;
  created_at: string;
}

export default function SwingsScreen() {
  const [swings, setSwings] = useState<Swing[]>([]);
  const [expandedIds, setExpandedIds] = useState<Set<string>>(new Set());
  const [loading, setLoading] = useState(true);
  const [downloadingId, setDownloadingId] = useState<string | null>(null);
  const [videoUrl, setVideoUrl] = useState<string | null>(null);
  const [videoModalVisible, setVideoModalVisible] = useState(false);

  useFocusEffect(
    useCallback(() => {
      fetchSwings();
    }, [])
  );

  const fetchSwings = async () => {
    setLoading(true);
    const { data: sessionData } = await supabase.auth.getSession();
    const userId = sessionData.session?.user?.id;

    if (!userId) {
      Alert.alert('Error', 'User not authenticated');
      return;
    }

    const { data, error } = await supabase
      .from('Swing')
      .select('*')
      .eq('user_id', userId)
      .order('created_at', { ascending: false });

    if (error) {
      console.error('Error fetching swings:', error);
      Alert.alert('Error', 'Could not load swings');
    } else {
      setSwings(data as Swing[]);
    }

    setLoading(false);
  };

  const getSignedUrl = async (s3Key: string): Promise<string | null> => {
    const response = await fetch(
      `http://192.168.1.186:8000/get-signed-url?s3_key=${encodeURIComponent(s3Key)}`
    );
    const json = await response.json();
    return response.ok && json.signed_url ? json.signed_url : null;
  };

  const handlePlay = async (s3Key: string) => {
    try {
      const url = await getSignedUrl(s3Key);
      if (!url) throw new Error('No signed URL');
      setVideoUrl(url);
      setVideoModalVisible(true);
    } catch (error) {
      console.error('❌ Error loading video:', error);
      Alert.alert('Error', 'Could not load video.');
    }
  };

  const handleDownload = async (s3Key: string, id: string) => {
    try {
      setDownloadingId(id);
      const filename = s3Key.split('/').pop() || 'swing.mp4';
      const fileUri = FileSystem.documentDirectory + filename;
      const signedUrl = await getSignedUrl(s3Key);
      if (!signedUrl) throw new Error('No signed URL');

      const downloadResumable = FileSystem.createDownloadResumable(signedUrl, fileUri);
      const result = await downloadResumable.downloadAsync();

      if (!result?.uri) throw new Error('Download failed');

      if (await Sharing.isAvailableAsync()) {
        await Sharing.shareAsync(result.uri);
      } else {
        Alert.alert('Download Complete', `Saved to ${result.uri}`);
      }
    } catch (err) {
      console.error('❌ Download error:', err);
      Alert.alert('Error', 'Could not download video.');
    } finally {
      setDownloadingId(null);
    }
  };

  const toggleAccordion = (id: string) => {
    const newSet = new Set(expandedIds);
    newSet.has(id) ? newSet.delete(id) : newSet.add(id);
    setExpandedIds(newSet);
  };

  return (
    <View style={styles.container}>
      <ThemedText type="title" style={styles.title}>Recent Swings</ThemedText>

      {loading ? (
        <ActivityIndicator size="large" color="#d5ead6" />
      ) : (
        <FlatList
          data={swings}
          keyExtractor={(item) => item.id}
          renderItem={({ item }) => (
            <SwingItem
              item={item}
              isExpanded={expandedIds.has(item.id)}
              onToggle={() => toggleAccordion(item.id)}
              onPlay={handlePlay}
              onDownload={handleDownload}
              downloadingId={downloadingId}
            />
          )}
          contentContainerStyle={styles.list}
        />
      )}

      <Modal
        isVisible={videoModalVisible}
        onBackdropPress={() => setVideoModalVisible(false)}
        style={{ margin: 0, justifyContent: 'center', alignItems: 'center' }}
      >
        <View style={styles.videoContainer}>
          {videoUrl && (
            <Video
              source={{ uri: videoUrl }}
              useNativeControls
              resizeMode={ResizeMode.CONTAIN}
              style={styles.video}
              shouldPlay
            />
          )}
        </View>
      </Modal>
    </View>
  );
}

const SwingItem = ({
  item,
  isExpanded,
  onToggle,
  onPlay,
  onDownload,
  downloadingId,
}: {
  item: Swing;
  isExpanded: boolean;
  onToggle: () => void;
  onPlay: (s3Key: string) => void;
  onDownload: (s3Key: string, id: string) => void;
  downloadingId: string | null;
}) => {
  const [thumbUrl, setThumbUrl] = useState<string | null>(null);

  useEffect(() => {
    const loadThumbnail = async () => {
      const key = `thumbnails/${item.uniqueid}.jpg`;
      const response = await fetch(
        `http://192.168.1.186:8000/get-signed-url?s3_key=${encodeURIComponent(key)}`
      );
      const json = await response.json();
      if (response.ok && json.signed_url) {
        setThumbUrl(json.signed_url);
      }
    };
    loadThumbnail();
  }, [item.uniqueid]);

  const date = new Date(item.created_at);
  const formattedDate = date.toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
  const formattedTime = date.toLocaleTimeString(undefined, {
    hour: 'numeric',
    minute: '2-digit',
  });

  return (
    <TouchableOpacity
      onPress={onToggle}
      activeOpacity={0.9}
      style={styles.card}
    >
      <View style={styles.gridRow}>
        <View style={styles.thumbColumn}>
          {thumbUrl ? (
            <Image source={{ uri: thumbUrl }} style={styles.thumbnail} />
          ) : (
            <View style={styles.thumbnailPlaceholder}>
              <Ionicons name="image-outline" size={24} color="#ccc" />
            </View>
          )}
        </View>

        <View style={styles.infoColumn}>
          <Text style={styles.date}>{formattedDate}</Text>
          <Text style={styles.time}>{formattedTime}</Text>
          <Text style={styles.idText}>ID: {item.uniqueid}</Text>
        </View>

        <View style={styles.actionsColumn}>
          <TouchableOpacity onPress={() => onPlay(item.s3_key)}>
            <Ionicons name="play-circle-outline" size={28} color="#fff" />
          </TouchableOpacity>

          <TouchableOpacity
            onPress={() => onDownload(item.s3_key, item.id)}
            style={{ marginTop: 8 }}
          >
            {downloadingId === item.id ? (
              <ActivityIndicator size="small" color="#ffffff" />
            ) : (
              <Ionicons name="download-outline" size={24} color="#fff" />
            )}
          </TouchableOpacity>
        </View>
      </View>

      {isExpanded && (
        <View style={styles.feedbackBox}>
          <Text style={styles.feedback}>{item.feedback}</Text>
        </View>
      )}
    </TouchableOpacity>
  );
};

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#082419',
    padding: 16,
    paddingTop: 40,
  },
  title: {
    fontSize: 26,
    fontWeight: 'bold',
    color: '#ffffff',
    textAlign: 'center',
    marginBottom: 16,
  },
  list: {
    paddingBottom: 30,
  },
  card: {
    backgroundColor: '#103826',
    borderRadius: 10,
    padding: 14,
    marginBottom: 12,
  },
  gridRow: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
  },
  thumbColumn: {
    width: 90,
  },
  infoColumn: {
    flex: 1,
    marginHorizontal: 12,
  },
  actionsColumn: {
    alignItems: 'center',
    justifyContent: 'center',
  },
  thumbnail: {
    width: 80,
    height: 45,
    borderRadius: 6,
    backgroundColor: '#0f1f18',
  },
  thumbnailPlaceholder: {
    width: 80,
    height: 45,
    borderRadius: 6,
    backgroundColor: '#1f1f1f',
    alignItems: 'center',
    justifyContent: 'center',
  },
  date: {
    color: '#d5ead6',
    fontSize: 14,
    fontWeight: 'bold',
  },
  time: {
    color: '#bfe6c8',
    fontSize: 12,
  },
  idText: {
    color: '#ffffff',
    fontSize: 13,
    marginTop: 4,
  },
  feedbackBox: {
    marginTop: 12,
    backgroundColor: '#154d35',
    padding: 10,
    borderRadius: 8,
  },
  feedback: {
    color: '#d5ead6',
    fontSize: 14,
  },
  videoContainer: {
    width: '90%',
    height: '60%',
    backgroundColor: '#000',
    borderRadius: 12,
    overflow: 'hidden',
  },
  video: {
    width: '100%',
    height: '100%',
  },
});
