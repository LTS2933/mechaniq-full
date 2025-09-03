// FeedbackModal.tsx
import React from 'react';
import { View, Text, StyleSheet, ScrollView, TouchableOpacity } from 'react-native';
import Modal from 'react-native-modal';

interface FeedbackModalProps {
  isVisible: boolean;
  onClose: () => void;
  feedback: string;
}

export default function FeedbackModal({ isVisible, onClose, feedback }: FeedbackModalProps) {
  const formattedFeedback = feedback.replace(/\\n/g, '\n'); // clean up formatting

  return (
    <Modal isVisible={isVisible} onBackdropPress={onClose}>
      <View style={styles.modalContainer}>
        <Text style={styles.title}>🎯 Swing Feedback</Text>
        <ScrollView style={styles.scrollView}>
          <Text style={styles.feedbackText}>{formattedFeedback}</Text>
        </ScrollView>
        <TouchableOpacity style={styles.button} onPress={onClose}>
          <Text style={styles.buttonText}>Got it!</Text>
        </TouchableOpacity>
      </View>
    </Modal>
  );
}

const styles = StyleSheet.create({
  modalContainer: {
    backgroundColor: '#fff',
    borderRadius: 16,
    padding: 20,
    maxHeight: '80%',
  },
  title: {
    fontSize: 20,
    fontWeight: 'bold',
    marginBottom: 12,
    textAlign: 'center',
  },
  scrollView: {
    marginBottom: 20,
  },
  feedbackText: {
    fontSize: 16,
    lineHeight: 24,
    color: '#333',
  },
  button: {
    backgroundColor: '#0066CC',
    paddingVertical: 12,
    borderRadius: 10,
    alignItems: 'center',
  },
  buttonText: {
    color: '#FFF',
    fontSize: 16,
    fontWeight: '600',
  },
});
