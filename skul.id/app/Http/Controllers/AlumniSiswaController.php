<?php

namespace App\Http\Controllers;

use Carbon\Carbon;
use App\Models\User;
use App\Models\Loker;
use App\Models\Artikel;
use App\Models\Pelatihan;
use App\Models\Sertifikasi;
use Illuminate\Http\Request;
use App\Models\AlumniSiswaProfile;
use Illuminate\Support\Facades\Log;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Cache;
use Illuminate\Support\Facades\Validator;

class AlumniSiswaController extends Controller
{
    public function index()
    {
        $user = User::with('alumniSiswaProfile')->find(Auth::id());
        $artikels = Artikel::latest()->take(4)->get(); // Ambil 4 artikel terbaru
        return view('alumni-siswa.beranda', compact('user', 'artikels'));
    }

    public function addProfile()
    {
        $allProvinsi = Http::get('https://www.emsifa.com/api-wilayah-indonesia/api/provinces.json')->json();

        // Filter hanya provinsi Sulawesi (ID 71-76)
        $provinsi = collect($allProvinsi)->filter(function ($item) {
            return in_array($item['id'], ['71', '72', '73', '74', '75', '76']);
        })->values();
        // Default: ambil kabupaten dari provinsi Sulawesi Selatan (id 73)
        $kabupaten = Http::get('https://www.emsifa.com/api-wilayah-indonesia/api/regencies/73.json')->json();

        $sekolah = Http::get("https://api-sekolah-indonesia.vercel.app/sekolah/SMK?page=1&perPage=100")->json();

        $user = Auth::user();

        // Jika profile sudah lengkap, redirect ke halaman utama alumni
        if ($user->profile_complete) {
            return redirect()->route('alumni-siswa.index');
        }

        $no_hp = Auth::user()->no_hp;

        return view('alumni-siswa.addProfile', compact('no_hp', 'provinsi', 'kabupaten', 'sekolah'));
    }

    public function cariSekolah(Request $request)
    {
        $query = strtolower(trim($request->get('q')));

        if (strlen($query) < 3) {
            return response()->json(['results' => []]);
        }

        $cacheKey = "cari_sekolah:$query";

        $results = Cache::remember($cacheKey, now()->addMinutes(60), function () use ($query) {
            try {
                $response = Http::timeout(3)->get("https://api-sekolah-indonesia.vercel.app/sekolah/s", [
                    'sekolah' => $query,
                ]);

                $data = $response->json();

                if (!isset($data['dataSekolah'])) {
                    return [];
                }

                return collect($data['dataSekolah'])
                    ->filter(fn($item) => isset($item['bentuk']) && in_array($item['bentuk'], ['SMK', 'SMA']))
                    ->map(fn($item) => [
                        'id' => $item['npsn'],
                        'text' => $item['sekolah'],
                    ])
                    ->take(10)
                    ->values()
                    ->toArray();
            } catch (\Exception $e) {
                return []; // fallback jika API timeout atau error
            }
        });

        return response()->json(['results' => $results]);
    }


    public function storeProfile(Request $request)
    {
        $validated = $request->validate([
            'nama_lengkap' => 'required|string',
            'nik' => 'required|string|unique:alumni_siswa_profiles,nik,' . Auth::id() . ',user_id',
            'tahun_kelulusan' => 'required|numeric',
            'jurusan_sekolah' => 'required|string',
            'asal_sekolah' => 'required|string',
            'no_telepon' => 'required|string',
            'jenis_kelamin' => 'required|string',
            'tanggal_lahir' => 'required|date',
            'alamat' => 'required|string',
            'status_saat_ini' => 'nullable|string',
            'bidang_pekerjaan' => 'required|string',
            'sertifikasi_terakhir' => 'nullable|string',
            'kesesuaian_sertifikasi' => 'nullable|string',
            'nama_universitas' => 'nullable|string',
            'jurusan_universitas' => 'nullable|string',
            'provinsi' => 'required|string',
            'kota' => 'required|string',
            'npsn' => 'required|string',
            'email' => 'required|email|unique:users,email,' . Auth::id(),
            'foto_profil' => 'nullable|image|mimes:jpg,jpeg,png|max:10048', // validasi file
            'skills' => 'required|string|max:255',
        ]);

        // Simpan/update profil
        $validatedProfile = $validated;
        unset($validatedProfile['email']); // email bukan bagian profile

        // Simpan file jika ada
        if ($request->hasFile('foto_profil')) {
            $file = $request->file('foto_profil');
            $file_name = $file->hashName(); // nama unik
            $file->move(public_path('storage/assets/profile'), $file_name);

            // Simpan nama file ke kolom foto_profil
            $validatedProfile['foto_profil'] = 'assets/profile/' . $file_name;
        }

        $validatedProfile['user_id'] = Auth::id();

        AlumniSiswaProfile::updateOrCreate(
            ['user_id' => Auth::id()],
            $validatedProfile
        );

        // Update user (email dan status profil)
        $user = User::find(Auth::id());
        $user->email = $validated['email'];
        $user->profile_complete = true;
        $user->save();

        return redirect()->route('alumni-siswa.index')->with('success', 'Profil berhasil disimpan!');
    }


    public function profile()
    {
        $user = User::with([
            'daftarSertifikasi.sertifikasi.mitra.mitraProfile',
            'daftarPelatihan.pelatihan.mitra.mitraProfile',
            'daftarLoker.loker.mitra.mitraProfile'
        ])->find(Auth::id());

        return view('alumni-siswa.profile', compact('user'));
    }


    public function editProfile()
    {
        $allProvinsi = Http::get('https://www.emsifa.com/api-wilayah-indonesia/api/provinces.json')->json();

        // Filter hanya provinsi Sulawesi (ID 71-76)
        $provinsi = collect($allProvinsi)->filter(function ($item) {
            return in_array($item['id'], ['71', '72', '73', '74', '75', '76']);
        })->values();
        // Default: ambil kabupaten dari provinsi Sulawesi Selatan (id 73)
        $kabupaten = Http::get('https://www.emsifa.com/api-wilayah-indonesia/api/regencies/73.json')->json();

        $sekolah = Http::get("https://api-sekolah-indonesia.vercel.app/sekolah/SMK?page=1&perPage=100")->json();

        $user = User::with('alumniSiswaProfile')->find(Auth::id());

        return view('alumni-siswa.editProfile', compact('user', 'provinsi', 'kabupaten', 'sekolah'));
    }

    public function updateProfile(Request $request)
    {
        $user = Auth::user();

        try {
            $validated = $request->validate([
                'nama_lengkap' => 'required|string',
                'nik' => 'required|string|unique:alumni_siswa_profiles,nik,' . $user->id . ',user_id',
                'kelas' => 'nullable|numeric',
                'tahun_kelulusan' => 'nullable|numeric',
                'jurusan_sekolah' => 'required|string',
                'asal_sekolah' => 'required|string',
                'no_hp' => 'required|string|unique:users,no_hp,' . $user->id,
                'jenis_kelamin' => 'required|string',
                'tanggal_lahir' => 'required|date',
                'alamat' => 'required|string',
                'status_saat_ini' => 'nullable|string',
                'bidang_pekerjaan' => 'required|string',
                'sertifikasi_terakhir' => 'nullable|string',
                'kesesuaian_sertifikasi' => 'nullable|string',
                'provinsi' => 'nullable|string',
                'kota' => 'nullable|string',
                'npsn' => 'nullable|string',
                'nama_universitas' => 'nullable|string',
                'jurusan_universitas' => 'nullable|string',
                'email' => 'required|email|unique:users,email,' . $user->id,
                'foto_profil' => 'nullable|image|mimes:jpg,jpeg,png|max:2048',
                'skills' => 'required|string|max:255',
            ]);

            // Update atau buat profil alumni
            $profile = AlumniSiswaProfile::firstOrNew(['user_id' => $user->id]);
            $profile->fill($validated);

            // Handle upload foto
            if ($request->hasFile('foto_profil')) {
                // hapus file lama jika ada
                if ($profile->foto_profil && file_exists(public_path('storage/' . $profile->foto_profil))) {
                    unlink(public_path('storage/' . $profile->foto_profil));
                }

                // upload file baru
                $file = $request->file('foto_profil');
                $fileName = $file->hashName();
                $file->move(public_path('storage/assets/profile'), $fileName);
                $profile->foto_profil = 'assets/profile/' . $fileName;
            }

            // Set default jika belum pernah upload
            if (!$profile->foto_profil) {
                $profile->foto_profil = 'assets/profile/user.png';
            }

            $profile->save();

            // Update user (email dan no hp)
            $user->update([
                'email' => $validated['email'],
            ]);

            return redirect()->route('alumni-siswa.profile')->with('success', 'Profil berhasil diperbarui!');
        } catch (\Throwable $e) {
            Log::error('Update profile gagal: ' . $e->getMessage());
            return redirect()->back()->withInput()->with('error', 'Terjadi kesalahan saat menyimpan data.');
        }
    }

    public function sertifikasi(Request $request)
    {
        $user = User::with('alumniSiswaProfile')->find(Auth::id());
        $totalAlumni = User::whereHas('alumniSiswaProfile')->count();

        $searchNama = $request->input('nama');
        $searchKota = $request->input('kota');
        $searchStatus = $request->input('status');

        $sertifikasi = Sertifikasi::query()
            ->whereDate('tanggal_mulai', '>', Carbon::today());

        if ($searchNama) {
            $sertifikasi->where('judul_sertifikasi', 'like', '%' . $searchNama . '%');
        }

        if ($searchKota) {
            $sertifikasi->where('kota', 'like', '%' . $searchKota . '%');
        }

        if ($searchStatus) {
            $sertifikasi->where('status', $searchStatus);
        }

        $sertifikasi = $sertifikasi->latest()->paginate(9)->withQueryString();
        $totalSertifikasi = Sertifikasi::whereDate('tanggal_mulai', '>', Carbon::today())->count();

        $totalAlumniBekerja = User::whereHas('alumniSiswaProfile', function ($query) {
            $query->where('status_saat_ini', 'Bekerja');
        })->count();

        return view('alumni-siswa.sertifikasi', compact('user', 'sertifikasi', 'totalAlumni', 'totalSertifikasi', 'totalAlumniBekerja', 'searchNama', 'searchKota', 'searchStatus'));
    }

    public function loker(Request $request)
    {
        $user = User::with('alumniSiswaProfile')->find(Auth::id());

        $query = Loker::query();

        // Filter hanya yang status = 'buka'
        $query->where('status', 'buka');

        if ($request->filled('search')) {
            $searchTerm = strtolower($request->search);

            $query->where(function ($q) use ($searchTerm) {
                $q->whereRaw('LOWER(posisi) LIKE ?', ["%{$searchTerm}%"])
                    ->orWhereRaw('LOWER(nama_perusahaan) LIKE ?', ["%{$searchTerm}%"]);
            });
        }

        if ($request->filled('tipe')) {
            $query->where('tipe', $request->tipe);
        }

        if ($request->filled('lokasi')) {
            $query->where('lokasi', $request->lokasi);
        }

        $loker = $query->paginate(9)->withQueryString();
        $lokasiList = Loker::where('status', 'buka')->select('lokasi')->distinct()->pluck('lokasi');

        $recommendations = [];

        try {
            if ($user && $user->alumniSiswaProfile) {
                $userId = $user->id;

                $response = Http::timeout(10)->get("http://127.0.0.1:8001/rekomendasi/{$userId}");

                if ($response->successful()) {
                    $data = $response->json();

                    // Pastikan hanya isi 'rekomendasi' yang dikirim ke view
                    $recommendations = $data['rekomendasi'] ?? [];
                }
            }
        } catch (\Exception $e) {
            Log::error("Gagal mengambil rekomendasi: " . $e->getMessage());
        }

        return view('alumni-siswa.loker', compact('user', 'loker', 'lokasiList', 'recommendations'));
    }


    public function pelatihan(Request $request)
    {
        $user = User::with('alumniSiswaProfile')->find(Auth::id());
        $totalAlumni = User::whereHas('alumniSiswaProfile')->count();

        $searchNama = $request->input('nama');
        $searchKota = $request->input('kota');
        $searchStatus = $request->input('status');

        $pelatihan = Pelatihan::query();

        // Filter hanya pelatihan yang belum dimulai
        $pelatihan->whereDate('tanggal_mulai', '>', Carbon::today());

        if ($searchNama) {
            $pelatihan->where('nama_pelatihan', 'like', '%' . $searchNama . '%');
        }

        if ($searchKota) {
            $pelatihan->where('kota', 'like', '%' . $searchKota . '%');
        }

        if ($searchStatus) {
            $pelatihan->where('status', $searchStatus);
        }

        $pelatihan = $pelatihan->latest()->paginate(12)->withQueryString();

        $totalPelatihan = Pelatihan::count();
        $pelatihanSelesai = Pelatihan::whereDate('tanggal_mulai', '<=', Carbon::today())->count();

        return view('alumni-siswa.pelatihan', compact(
            'user',
            'pelatihan',
            'totalAlumni',
            'totalPelatihan',
            'pelatihanSelesai',
            'searchNama',
            'searchKota',
            'searchStatus'
        ));
    }

    public function ikatan(Request $request)
    {
        $user = User::with('alumniSiswaProfile')->find(Auth::id());

        $alumni = User::where('id', '!=', Auth::id())
            ->whereHas('alumniSiswaProfile', function ($query) use ($user) {
                $query->where('asal_sekolah', $user->alumniSiswaProfile->asal_sekolah);
            })
            ->with('alumniSiswaProfile');

        if ($request->filled('search')) {
            $search = $request->input('search');
            $alumni->whereHas('alumniSiswaProfile', function ($query) use ($search) {
                $query->where('nama_lengkap', 'like', '%' . $search . '%');
            });
        }

        $alumni = $alumni->paginate(9)->withQueryString();

        return view('alumni-siswa.ikatan', compact('user', 'alumni'));
    }

    public function kuliah()
    {
        $user = User::with('alumniSiswaProfile')->find(Auth::id());
        return view('alumni-siswa.kuliah', compact('user'));
    }

    public function artikel()
    {
        $user = User::with('alumniSiswaProfile')->find(Auth::id());
        $artikels = Artikel::latest()->paginate(6)->withQueryString();

        return view('alumni-siswa.artikel', compact('user', 'artikels'));
    }

    public function lokerSearch(Request $request)
    {
        $user = User::with('alumniSiswaProfile')->find(Auth::id());

        $query = Loker::query();

        if ($request->filled('search')) {
            $searchTerm = strtolower($request->search);

            $query->where(function ($q) use ($searchTerm) {
                $q->whereRaw('LOWER(posisi) LIKE ?', ["%{$searchTerm}%"])
                    ->orWhereRaw('LOWER(nama_perusahaan) LIKE ?', ["%{$searchTerm}%"]);
            });
        }


        if ($request->filled('tipe')) {
            $query->where('tipe', $request->tipe);
        }

        if ($request->filled('lokasi')) {
            $query->where('lokasi', $request->lokasi);
        }

        $loker = $query->paginate(9)->withQueryString();
        $lokasiList = Loker::select('lokasi')->distinct()->pluck('lokasi');

        return view('alumni-siswa.loker', compact('user', 'loker', 'lokasiList'));
    }
}
