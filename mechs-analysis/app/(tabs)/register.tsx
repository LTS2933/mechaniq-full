import { ThemedText } from "@/components/ThemedText";
import { supabase } from "@/lib/supabase";
import { useRouter } from "expo-router";
import { useState, useEffect } from "react";
import {
  ActivityIndicator,
  Dimensions,
  KeyboardAvoidingView,
  Keyboard,
  Platform,
  Pressable,
  ScrollView,
  StyleSheet,
  TextInput,
  TouchableWithoutFeedback,
  View,
} from "react-native";
import Animated, {
  useSharedValue,
  useAnimatedStyle,
  withRepeat,
  withTiming,
  Easing,
} from "react-native-reanimated";
import { LinearGradient } from "expo-linear-gradient";

const { width: SCREEN_WIDTH } = Dimensions.get("window");

export default function RegisterScreen() {
  const router = useRouter();
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const glow = useSharedValue(0.4);
  const surge = useSharedValue(-400);

  useEffect(() => {
    if (!loading) {
      glow.value = withRepeat(withTiming(1, { duration: 1500, easing: Easing.inOut(Easing.ease) }), -1, true);
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

  const handleRegister = async () => {
    setLoading(true);
    setError(null);

    const { data: existingUser, error: checkUserError } = await supabase
      .from("User")
      .select("id")
      .eq("username", username)
      .maybeSingle();

    if (checkUserError) {
      setLoading(false);
      setError("Error checking username. Please try again.");
      return;
    }
    if (existingUser) {
      setLoading(false);
      setError("Username already taken.");
      return;
    }

    const { data: existingEmail, error: checkEmailError } = await supabase
      .from("User")
      .select("id")
      .ilike("email", email)
      .maybeSingle();

    if (checkEmailError) {
      setLoading(false);
      setError("Error checking email. Please try again.");
      return;
    }
    if (existingEmail) {
      setLoading(false);
      setError("Email already in use.");
      return;
    }

    const { data, error: signUpError } = await supabase.auth.signUp({
      email,
      password,
      options: {
        data: {
          full_name: fullName,
          username: username,
        },
      },
    });

    if (signUpError) {
      setLoading(false);
      setError(signUpError.message);
      return;
    }

    if (data.user) {
      const { error: insertError } = await supabase.from("User").insert([
        {
          username,
          email,
          full_name: fullName,
        },
      ]);
      if (insertError) {
        setLoading(false);
        setError(insertError.message);
        return;
      }
    }

    setLoading(false);
    router.replace("/(tabs)/upload");
  };

  return (
    <TouchableWithoutFeedback onPress={Keyboard.dismiss}>
      <KeyboardAvoidingView
        style={{ flex: 1 }}
        behavior={Platform.OS === "ios" ? "padding" : "height"}
        keyboardVerticalOffset={Platform.OS === "ios" ? 0 : 20}
      >
        <View style={styles.container}>
          <View style={styles.bg} pointerEvents="none">
            <LinearGradient
              colors={["#0f1f18", "#123224", "#0f1f18"]}
              start={{ x: 0, y: 0 }}
              end={{ x: 1, y: 1 }}
              style={StyleSheet.absoluteFillObject}
            />
          </View>

          <ScrollView
            contentContainerStyle={{ paddingBottom: 40, paddingHorizontal: 24, flexGrow: 1, justifyContent: "center" }}
            keyboardShouldPersistTaps="handled"
            showsVerticalScrollIndicator={false}
          >
            <View style={styles.card}>
              <ThemedText type="title" style={styles.title}>Create Account</ThemedText>

              <View style={styles.inputGroup}>
                <ThemedText style={styles.label}>Full Name</ThemedText>
                <TextInput
                  style={styles.input}
                  placeholder="Full Name"
                  value={fullName}
                  onChangeText={setFullName}
                  autoCapitalize="words"
                />
              </View>
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
                <ThemedText style={styles.label}>Email</ThemedText>
                <TextInput
                  style={styles.input}
                  placeholder="Email"
                  value={email}
                  onChangeText={setEmail}
                  autoCapitalize="none"
                  keyboardType="email-address"
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
                <ThemedText style={styles.error}>
                  {error}
                </ThemedText>
              )}

              <Animated.View style={[styles.glowWrapper, glowStyle]}>
                <Pressable
                  style={[styles.button, loading && styles.disabledButton]}
                  onPress={handleRegister}
                  disabled={loading}
                >
                  <ThemedText style={styles.buttonText}>
                    {loading ? "Creating..." : "Register"}
                  </ThemedText>

                  {loading && (
                    <Animated.View style={[styles.surgeBar, surgeStyle]}>
                      <LinearGradient
                        colors={["transparent", "#ffffff66", "transparent"]}
                        start={{ x: 0, y: 0 }}
                        end={{ x: 1, y: 0 }}
                        style={styles.surgeLine}
                      />
                    </Animated.View>
                  )}
                </Pressable>
              </Animated.View>
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
    backgroundColor: "#082419",
  },
  bg: {
    ...StyleSheet.absoluteFillObject,
    zIndex: -1,
  },
  card: {
    backgroundColor: "#1a2e25",
    borderRadius: 20,
    padding: 24,
    borderWidth: 2,
    borderColor: "#ffffff33",
    shadowColor: "#000",
    shadowOffset: { width: 0, height: 10 },
    shadowOpacity: 0.2,
    shadowRadius: 20,
    elevation: 6,
  },
  title: {
    fontSize: 26,
    fontWeight: "900",
    color: "#ffffff",
    letterSpacing: 2,
    textAlign: "center",
    marginBottom: 16,
    textShadowColor: "#f44336",
    textShadowOffset: { width: 0, height: 0 },
    textShadowRadius: 10,
  },
  inputGroup: {
    marginBottom: 16,
  },
  label: {
    color: "#e0f2f1",
    fontWeight: "600",
    fontSize: 15,
    marginBottom: 6,
  },
  input: {
    height: 44,
    backgroundColor: "#243c32",
    borderRadius: 10,
    paddingHorizontal: 12,
    color: "#fff",
    borderWidth: 1,
    borderColor: "#ffffff22",
    fontSize: 15,
  },
  error: {
    color: "#f88",
    fontSize: 14,
    fontWeight: "600",
    textAlign: "center",
    marginTop: 6,
  },
  button: {
    backgroundColor: "#f44336",
    paddingVertical: 16,
    borderRadius: 16,
    alignItems: "center",
    marginTop: 16,
    borderWidth: 2,
    borderColor: "#ffffff88",
    shadowColor: "#f44336",
    shadowOffset: { width: 0, height: 6 },
    shadowOpacity: 0.3,
    shadowRadius: 12,
    elevation: 6,
    overflow: "hidden",
  },
  disabledButton: {
    opacity: 0.65,
  },
  buttonText: {
    color: "#fff",
    fontSize: 16,
    fontWeight: "900",
    letterSpacing: 1,
    textTransform: "uppercase",
  },
  glowWrapper: {
    shadowColor: "#ffffff",
    shadowOffset: { width: 0, height: 0 },
    shadowRadius: 18,
  },
  surgeBar: {
    position: "absolute",
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
