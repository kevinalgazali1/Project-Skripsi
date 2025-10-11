<?php

namespace App\Http\Controllers;

use Exception;
use App\Models\User;
use App\Models\MitraProfile;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Auth;
use Illuminate\Support\Facades\Http;

class MitraController extends Controller
{
    public function addProfile()
    {
        // Ambil daftar provinsi dari API
        $provinsi = Http::get('https://www.emsifa.com/api-wilayah-indonesia/api/provinces.json')->json();

        // Default: ambil kabupaten dari provinsi Sulawesi Selatan (id 73)
        $kabupaten = Http::get('https://www.emsifa.com/api-wilayah-indonesia/api/regencies/73.json')->json();

        $user = Auth::user();

        // Jika profile sudah lengkap, redirect ke halaman utama alumni
        if ($user->profile_complete) {
            return redirect()->route('alumni-siswa.index');
        }

        $no_hp = Auth::user()->no_hp;

        return view('mitra.addProfile', compact('provinsi', 'kabupaten', 'no_hp'));
    }

    public function storeProfile(Request $request)
    {
        $validated = $request->validate([
            'nama_instansi'     => 'required|string|max:255',
            'penanggung_jawab'    => 'required|string|max:255',
            'kategori'            => 'required|string',
            'alamat'              => 'required|string',
            'provinsi'            => 'required|string',
            'kota'                => 'required|string',
            'bidang_industri'     => 'required|string',
            'email' => 'required|email|unique:users,email,' . Auth::id(),
        ]);

        // Simpan/update profil
        $validatedProfile = $validated;
        unset($validatedProfile['email']); // email bukan bagian profile, hapus supaya tidak masuk ke profile

        $validatedProfile['user_id'] = Auth::id();

        MitraProfile::updateOrCreate(
            ['user_id' => Auth::id()],
            $validatedProfile
        );

        // Update user (termasuk email dan profile_complete)
        $user = User::find(Auth::id());
        $user->email = $validated['email'];
        $user->profile_complete = true;
        $user->save();

        return redirect()->route('mitra.index')->with('success', 'Profil berhasil disimpan!');
    }

    public function index()
    {
        $user = User::with('mitraProfile')->find(Auth::id());
        return view('mitra.beranda', compact('user'));
    }

    public function profile()
    {
        // Ambil daftar provinsi dari API
        $provinsi = Http::get('https://www.emsifa.com/api-wilayah-indonesia/api/provinces.json')->json();

        // Default: ambil kabupaten dari provinsi Sulawesi Selatan (id 73)
        $kota = Http::get('https://www.emsifa.com/api-wilayah-indonesia/api/regencies/73.json')->json();

        $user = Auth::user();
        return view('mitra.profile', compact('user', 'provinsi', 'kota'));
    }

    public function updateProfile(Request $request)
    {
        try {
            $data = $request->all();

            // Ambil daftar provinsi dari API
            $provinsiList = Http::get('https://www.emsifa.com/api-wilayah-indonesia/api/provinces.json')->json();

            // Cari nama provinsi berdasarkan ID
            $provinsiName = collect($provinsiList)->firstWhere('id', $request->provinsi)['name'] ?? null;

            if (!$provinsiName) {
                return redirect()->back()->with('error', 'Gagal memperbarui: Provinsi tidak valid.');
            }

            // Update profil
            $user = Auth::user();
            $user->mitraProfile->update([
                'provinsi' => $provinsiName,
                'kota' => $request->kota,
                'alamat' => $request->alamat,
                'nama_instansi' => $request->nama_instansi,
                'penanggung_jawab' => $request->penanggung_jawab,
                'bidang_industri' => $request->bidang_industri,
                'kategori' => $request->kategori,
            ]);

            return redirect()->back()->with('success', 'Profil mitra berhasil diperbarui');
        } catch (Exception $e) {
            // Log error jika perlu: Log::error($e);
            return redirect()->back()->with('error', 'Terjadi kesalahan saat menyimpan data');
        }
    }
}
