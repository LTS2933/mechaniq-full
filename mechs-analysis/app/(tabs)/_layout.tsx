// app/(tabs)/_layout.tsx
import { Tabs } from 'expo-router';
import { Ionicons, MaterialCommunityIcons, Feather, FontAwesome5 } from '@expo/vector-icons';
import { useColorScheme } from 'react-native';

export default function TabLayout() {
  const themeColor = '#1a2e25';
  const activeTint = '#f44336'; // red accent
  const inactiveTint = '#a5d6a7';

  return (
    <Tabs
      screenOptions={{
        headerShown: false,
        tabBarStyle: {
          backgroundColor: themeColor,
          borderTopWidth: 0,
          height: 85,
          paddingBottom: 20,
          paddingTop: 6,
        },
        tabBarActiveTintColor: activeTint,
        tabBarInactiveTintColor: inactiveTint,
        tabBarLabelStyle: {
          fontSize: 10,
          fontWeight: '600',
        },
      }}
    >
      <Tabs.Screen
        name="upload"
        options={{
          title: 'Upload',
          tabBarIcon: ({ color, size }) => (
            <Feather name="upload-cloud" size={size} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="swings"
        options={{
          title: 'History',
          tabBarIcon: ({ color, size }) => (
            <MaterialCommunityIcons name="history" size={size} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="faq"
        options={{
          title: 'Guide',
          tabBarIcon: ({ color, size }) => (
            <FontAwesome5 name="question-circle" size={size} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="drills"
        options={{
          title: 'Drills',
          tabBarIcon: ({ color, size }) => (
            <FontAwesome5 name="dumbbell" size={size} color={color} />
          ),
        }}
      />
      <Tabs.Screen
        name="profile"
        options={{
          title: 'Profile',
          tabBarIcon: ({ color, size }) => (
            <Ionicons name="person-circle-outline" size={size} color={color} />
          ),
        }}
      />
    </Tabs>
  );
}
