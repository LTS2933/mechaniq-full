import { createClient } from '@supabase/supabase-js';

const supabaseUrl = 'https://qzjvoetjuomllkxbsrvg.supabase.co';
const supabaseKey = process.env.EXPO_PUBLIC_SUPABASE_KEY;
if (!supabaseKey) {
  throw new Error('Missing Supabase key in environment variables');
}
export const supabase = createClient(supabaseUrl, supabaseKey);
