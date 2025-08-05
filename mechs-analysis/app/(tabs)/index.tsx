import { ThemedText } from "@/components/ThemedText";
import { supabase } from "@/lib/supabase";
import { Image } from "expo-image";
import { router } from "expo-router";
import { useState } from "react";
import {
  ActivityIndicator,
  Dimensions,
  KeyboardAvoidingView,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  TextInput,
  View
} from "react-native";

// Get screen width for responsiveness
const SCREEN_WIDTH = Dimensions.get("window").width;

export default function HomeScreen() {
  const [isSignedIn, setIsSignedIn] = useState(false);
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSignIn = async () => {
    setLoading(true);
    setError(null);
    const { data: userRecord, error: userCheckError } = await supabase
      .from("User")
      .select("email")
      .eq("username", username)
      .maybeSingle();

    if (userCheckError) {
      setLoading(false);
      setError("Error checking user. Please try again.");
      return;
    }
    if (!userRecord) {
      setLoading(false);
      setError("User not found.");
      return;
    }

    const { error: signInError } = await supabase.auth.signInWithPassword({
      email: userRecord.email,
      password,
    });

    setLoading(false);
    if (signInError) {
      setError("Incorrect password.");
    } else {
      setIsSignedIn(true);
    }
  };

  const handleSignOut = async () => {
    await supabase.auth.signOut();
    setIsSignedIn(false);
    setUsername("");
    setPassword("");
  };

  // ---- LOGIN UI ----
  if (!isSignedIn) {
    return (
    <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        keyboardVerticalOffset={Platform.OS === "ios" ? 0 : 20}
      >
      
      <ScrollView
        style={styles.bg}
        contentContainerStyle={styles.centerContent}
        keyboardShouldPersistTaps="handled"
      >
        <View style={styles.logoArea}>
          <Image source={require("@/assets/logo.png")} style={styles.logo} />
          <ThemedText type="title" style={styles.appTitle}>
            Baseball Coach AI
          </ThemedText>
          <ThemedText style={styles.appSubtitle}>
            Perfect your swing and pitch with AI-powered analysis
          </ThemedText>
        </View>

        <View style={styles.loginCard}>
          <ThemedText type="subtitle" style={styles.loginHeader}>
            Sign In to Get Started
          </ThemedText>
          <View style={styles.inputGroup}>
            <ThemedText style={styles.inputLabel}>Username</ThemedText>
            <TextInput
              style={styles.input}
              placeholder="Enter your username"
              autoCapitalize="none"
              autoCorrect={false}
              value={username}
              onChangeText={setUsername}
              editable={!loading}
            />
          </View>
          <View style={styles.inputGroup}>
            <ThemedText style={styles.inputLabel}>Password</ThemedText>
            <TextInput
              style={styles.input}
              placeholder="Enter your password"
              secureTextEntry
              autoCapitalize="none"
              autoCorrect={false}
              value={password}
              onChangeText={setPassword}
              editable={!loading}
            />
          </View>
          {error ? (
            <ThemedText style={styles.errorText}>{error}</ThemedText>
          ) : null}
          <Pressable
            style={({ pressed }) => [
              styles.signInButton,
              pressed && { opacity: 0.7 },
            ]}
            onPress={loading ? undefined : handleSignIn}
          >
            {loading ? (
              <ActivityIndicator color="#fff" />
            ) : (
              <ThemedText style={styles.buttonText}>Sign In</ThemedText>
            )}
          </Pressable>
          <Pressable
            style={({ pressed }) => [
              styles.createAccountButton,
              pressed && { opacity: 0.7 },
            ]}
            onPress={() => router.push("/(tabs)/register")}
          >
            <ThemedText style={styles.createAccountText}>
              Create Account
            </ThemedText>
          </Pressable>
        </View>
      </ScrollView>
      </KeyboardAvoidingView>
    );
  }

  // ---- DASHBOARD UI ----
  return (
    <ScrollView
      style={styles.bg}
      contentContainerStyle={styles.centerContent}
      keyboardShouldPersistTaps="handled"
    >
      <View style={styles.dashboardCard}>
        <View style={styles.dashboardHeader}>
          <ThemedText type="title" style={styles.dashboardTitle}>
            Welcome Back!
          </ThemedText>
          <Pressable style={styles.signOutButton} onPress={handleSignOut}>
            <ThemedText style={styles.signOutText}>Sign Out</ThemedText>
          </Pressable>
        </View>
        <ThemedText type="subtitle" style={styles.dashboardSubtitle}>
          Your Progress
        </ThemedText>
        <View style={styles.statsGroup}>
          <View style={styles.statCard}>
            <ThemedText style={styles.statIcon}>📹</ThemedText>
            <ThemedText style={styles.statNum}>12</ThemedText>
            <ThemedText style={styles.statLabel}>Videos Analyzed</ThemedText>
          </View>
          <View style={styles.statCard}>
            <ThemedText style={styles.statIcon}>⭐</ThemedText>
            <ThemedText style={styles.statNum}>8.5</ThemedText>
            <ThemedText style={styles.statLabel}>Avg Rating</ThemedText>
          </View>
        </View>
      </View>
    </ScrollView>
  );
}

// ---- STYLES ----
const styles = StyleSheet.create({
  bg: {
    flex: 1,
    backgroundColor: "#f4f8fc",
  },
  centerContent: {
    alignItems: "center",
    justifyContent: "flex-start",
    paddingTop: 42,
    paddingBottom: 42,
    minHeight: "100%",
  },
  logoArea: {
    alignItems: "center",
    marginBottom: 28,
    gap: 8,
  },
  logo: {
    width: 86,
    height: 86,
    marginBottom: 12,
    borderRadius: 18,
    backgroundColor: "#fff",
  },
  appTitle: {
    marginBottom: 6,
    fontSize: 25,
    fontWeight: "bold",
    color: "#1850B5",
    letterSpacing: 0.2,
  },
  appSubtitle: {
    color: "#4a6480",
    opacity: 0.78,
    fontSize: 15,
    textAlign: "center",
    paddingHorizontal: 12,
    marginBottom: 8,
  },
  loginCard: {
    width: SCREEN_WIDTH < 400 ? "100%" : 340,
    backgroundColor: "#fff",
    borderRadius: 18,
    padding: 26,
    marginHorizontal: 12,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 5 },
    shadowOpacity: 0.06,
    shadowRadius: 20,
    elevation: 5,
    alignItems: "stretch",
  },
  loginHeader: {
    textAlign: "center",
    marginBottom: 18,
    fontSize: 19,
    fontWeight: "600",
    color: "#1a2541",
  },
  inputGroup: {
    marginBottom: 17,
  },
  inputLabel: {
    marginBottom: 5,
    fontSize: 15,
    color: "#1a2541",
    fontWeight: "500",
  },
  input: {
    height: 44,
    borderWidth: 1,
    borderColor: "#e2e7ef",
    borderRadius: 10,
    paddingHorizontal: 12,
    backgroundColor: "#f7f9fc",
    fontSize: 15,
  },
  errorText: {
    color: "#e33",
    marginBottom: 9,
    textAlign: "center",
    fontSize: 14,
    fontWeight: "600",
  },
  signInButton: {
    height: 44,
    borderRadius: 10,
    backgroundColor: "#2267D5",
    alignItems: "center",
    justifyContent: "center",
    marginBottom: 14,
    marginTop: 3,
  },
  buttonText: {
    color: "white",
    fontSize: 17,
    fontWeight: "700",
    letterSpacing: 0.15,
  },
  createAccountButton: {
    borderWidth: 1.5,
    borderColor: "#2267D5",
    borderRadius: 10,
    height: 44,
    alignItems: "center",
    justifyContent: "center",
  },
  createAccountText: {
    color: "#2267D5",
    fontSize: 16,
    fontWeight: "bold",
    letterSpacing: 0.1,
  },
  // ---- Dashboard ----
  dashboardCard: {
    width: SCREEN_WIDTH < 400 ? "100%" : 350,
    backgroundColor: "#fff",
    borderRadius: 18,
    padding: 28,
    marginHorizontal: 12,
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 5 },
    shadowOpacity: 0.08,
    shadowRadius: 22,
    elevation: 7,
    alignItems: "stretch",
  },
  dashboardHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 16,
    gap: 10,
  },
  dashboardTitle: {
    fontSize: 22,
    fontWeight: "700",
    color: "#1A2541",
  },
  signOutButton: {
    borderWidth: 1.5,
    borderColor: "#2267D5",
    backgroundColor: "#f4f8fc",
    borderRadius: 18,
    paddingVertical: 7,
    paddingHorizontal: 19,
    marginLeft: 10,
  },
  signOutText: {
    color: "#2267D5",
    fontWeight: "bold",
    fontSize: 15,
    letterSpacing: 0.1,
  },
  dashboardSubtitle: {
    fontSize: 16,
    color: "#1a2541",
    fontWeight: "500",
    marginBottom: 18,
    marginTop: 2,
  },
  statsGroup: {
    flexDirection: SCREEN_WIDTH < 350 ? "column" : "row",
    justifyContent: "space-between",
    gap: 15,
  },
  statCard: {
    flex: 1,
    backgroundColor: "#f8fbff",
    borderRadius: 13,
    alignItems: "center",
    justifyContent: "center",
    paddingVertical: 24,
    marginHorizontal: 4,
    shadowColor: "#2267D5",
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: Platform.OS === "ios" ? 0.08 : 0.13,
    shadowRadius: 10,
    elevation: 4,
    minWidth: 110,
  },
  statIcon: {
    fontSize: 32,
    marginBottom: 4,
  },
  statNum: {
    fontSize: 28,
    color: "#2267D5",
    fontWeight: "bold",
    marginBottom: 2,
    marginTop: 1,
  },
  statLabel: {
    fontSize: 14,
    color: "#43618a",
    opacity: 0.82,
    textAlign: "center",
    marginTop: 2,
  },
});
