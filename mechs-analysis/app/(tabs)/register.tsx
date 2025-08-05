import { ThemedText } from "@/components/ThemedText";
import { ThemedView } from "@/components/ThemedView";
import { supabase } from "@/lib/supabase";
import { useRouter } from "expo-router";
import { useState } from "react";
import { ActivityIndicator, StyleSheet, TextInput, View } from "react-native";

export default function RegisterScreen() {
  const router = useRouter();
  const [fullName, setFullName] = useState("");
  const [email, setEmail] = useState("");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

    const handleRegister = async () => {
    setLoading(true);
    setError(null);

    // 1. Check if username already exists in 'User' table
    const { data: existingUser, error: checkUserError } = await supabase
        .from('User')
        .select('id')
        .eq('username', username)
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

    // 2. Check if email already exists in 'User' table
    //console.log("Checking email:", email);
    const { data: existingEmail, error: checkEmailError } = await supabase
        .from('User')
        .select('id')
        .ilike('email', email)
        .maybeSingle();
        //console.log("existingEmail:", existingEmail, "checkEmailError:", checkEmailError);

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

    // 3. Register user with Supabase Auth
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

    // 4. Insert into 'User' table (do NOT include password or password_hash)
    if (data.user) {
        const { error: insertError } = await supabase
        .from('User')
        .insert([{
            username,
            email,
            full_name: fullName,
        }]);
        if (insertError) {
        setLoading(false);
        setError(insertError.message);
        return;
        }
    }

    setLoading(false);
    router.replace("/(tabs)");
    };

  return (
    <ThemedView style={styles.container}>
      <ThemedText type="title" style={styles.title}>Create Account</ThemedText>
      <View style={styles.inputContainer}>
        <ThemedText style={styles.inputLabel}>Full Name</ThemedText>
        <TextInput
          style={styles.input}
          placeholder="Full Name"
          value={fullName}
          onChangeText={setFullName}
          autoCapitalize="words"
        />
      </View>
      <View style={styles.inputContainer}>
        <ThemedText style={styles.inputLabel}>Username</ThemedText>
        <TextInput
          style={styles.input}
          placeholder="Username"
          value={username}
          onChangeText={setUsername}
          autoCapitalize="none"
        />
      </View>
      <View style={styles.inputContainer}>
        <ThemedText style={styles.inputLabel}>Email</ThemedText>
        <TextInput
          style={styles.input}
          placeholder="Email"
          value={email}
          onChangeText={setEmail}
          autoCapitalize="none"
          keyboardType="email-address"
        />
      </View>
      <View style={styles.inputContainer}>
        <ThemedText style={styles.inputLabel}>Password</ThemedText>
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
        <ThemedText style={{ color: "red", marginBottom: 8, textAlign: "center" }}>
          {error}
        </ThemedText>
      )}
      <View style={styles.button} pointerEvents={loading ? "none" : "auto"}>
        <ThemedText
          style={styles.buttonText}
          onPress={loading ? undefined : handleRegister}
        >
          {loading ? <ActivityIndicator color="#fff" /> : "Register"}
        </ThemedText>
      </View>
    </ThemedView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, justifyContent: "center", padding: 24 },
  title: { textAlign: "center", marginBottom: 24 },
  inputContainer: { marginBottom: 16, maxWidth: 320, width: "100%", alignSelf: "center" },
  inputLabel: { marginBottom: 8, fontSize: 16 },
  input: {
    height: 40,
    borderWidth: 1,
    borderColor: "#ddd",
    borderRadius: 8,
    paddingHorizontal: 10,
    backgroundColor: "#f9f9f9",
    width: "100%",
  },
  button: {
    height: 40,
    borderRadius: 8,
    justifyContent: "center",
    alignItems: "center",
    marginTop: 16,
    backgroundColor: "#007AFF",
    width: "100%",
    maxWidth: 320,
    alignSelf: "center",
  },
  buttonText: { color: "#fff", fontSize: 16, fontWeight: "bold" },
});